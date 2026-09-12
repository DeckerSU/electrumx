# Copyright (c) 2026, the ElectrumX authors

"""Regression vectors for post-Sapling Zcash transaction encodings."""

import json
from collections import defaultdict
from pathlib import Path

import pytest

from electrumx.lib.coins import Zcash
from electrumx.lib.hash import double_sha256, hash_to_hex_str, hex_str_to_hash
from electrumx.lib.tx import DeserializerZcash, TxZcash, ZcashDeserializeError
from electrumx.server.mempool import MemPool, MemPoolTx


FIXTURES = sorted((Path(__file__).parents[1] / 'transactions').glob('zcash_*.json'))
BLOCK_FIXTURES = [
    path for path in sorted((Path(__file__).parents[1] / 'blocks').glob('zcash_mainnet_*.json'))
    if 'merkle_root' in json.loads(path.read_text())
]


@pytest.mark.parametrize('fixture_path', FIXTURES, ids=lambda path: path.stem)
def test_zcash_v5_v6_transaction(fixture_path):
    fixture = json.loads(fixture_path.read_text())
    raw_tx = bytes.fromhex(fixture['hex'])

    deserializer = DeserializerZcash(raw_tx)
    tx = deserializer.read_tx()

    assert isinstance(tx, TxZcash)
    assert deserializer.cursor == len(raw_tx)
    assert hash_to_hex_str(tx.txid_rev) == fixture['txid']
    assert tx.wtxid_rev == tx.txid_rev
    with pytest.raises(ValueError, match='serializing'):
        tx.serialize()
    assert tx.version == fixture['version']
    assert tx.version_group_id == fixture['version_group_id']
    assert tx.consensus_branch_id == fixture['consensus_branch_id']
    assert tx.locktime == fixture['locktime']
    assert tx.expiry_height == fixture['expiry_height']
    assert tx.fee_adjustment == fixture['fee_adjustment_zat']

    assert len(tx.inputs) == len(fixture['inputs'])
    for expected, actual in zip(fixture['inputs'], tx.inputs):
        assert actual.sequence == expected['sequence']
        if expected['coinbase'] is not None:
            assert actual.is_generation()
            assert actual.script.hex() == expected['coinbase']
        else:
            assert hash_to_hex_str(actual.prev_txid_rev) == expected['txid']
            assert actual.prev_idx == expected['vout']
            assert actual.script.hex() == expected['script']

    assert [(output.value, output.pk_script.hex()) for output in tx.outputs] == [
        (output['value_zat'], output['pk_script']) for output in fixture['outputs']
    ]
    _, size = DeserializerZcash(raw_tx).read_tx_and_vsize()
    assert size == len(raw_tx)


def test_zcash_transaction_respects_start_offset():
    fixture = json.loads(FIXTURES[0].read_text())
    raw_tx = bytes.fromhex(fixture['hex'])
    deserializer = DeserializerZcash(b'\x00' + raw_tx, start=1)
    tx = deserializer.read_tx()

    assert hash_to_hex_str(tx.txid_rev) == fixture['txid']
    assert deserializer.cursor == len(raw_tx) + 1


@pytest.mark.parametrize('fixture_path', BLOCK_FIXTURES, ids=lambda path: path.stem)
def test_zcash_block_consumption_and_merkle_root(fixture_path):
    fixture = json.loads(fixture_path.read_text())
    raw_block = bytes.fromhex(fixture['block'])
    header = Zcash.block_header(raw_block, fixture['height'])
    deserializer = DeserializerZcash(raw_block, start=len(header))
    transactions = deserializer.read_tx_block()

    assert deserializer.cursor == len(raw_block)
    if fixture['height'] == 3428144:
        assert [tx.version for tx in transactions] == [6, 4, 6]
    hashes = [tx.txid_rev for tx in transactions]
    while len(hashes) > 1:
        if len(hashes) & 1:
            hashes.append(hashes[-1])
        hashes = [double_sha256(left + right) for left, right in zip(hashes[::2], hashes[1::2])]
    assert hashes[0] == hex_str_to_hash(fixture['merkle_root'])


def test_zcash_v5_v6_rejects_truncated_data():
    fixture = json.loads(FIXTURES[-1].read_text())
    with pytest.raises(ZcashDeserializeError, match='truncated'):
        DeserializerZcash(bytes.fromhex(fixture['hex'])[:-1]).read_tx()


def test_zcash_v5_v6_rejects_invalid_header_and_compact_size():
    fixture = json.loads(FIXTURES[0].read_text())
    raw_tx = bytes.fromhex(fixture['hex'])

    wrong_group = raw_tx[:4] + b'\x00' * 4 + raw_tx[8:]
    with pytest.raises(ZcashDeserializeError, match='unknown Zcash transaction format'):
        DeserializerZcash(wrong_group).read_tx()

    # The first CompactSize (the transparent input count) is at offset 20.
    noncanonical = raw_tx[:20] + b'\xfd\x01\x00' + raw_tx[21:]
    with pytest.raises(ZcashDeserializeError, match='non-canonical'):
        DeserializerZcash(noncanonical).read_tx()


@pytest.mark.parametrize('name', ('zcash_mainnet_f9369e', 'zcash_mainnet_f15bfc'))
def test_zcash_effecting_and_authorizing_bytes_have_expected_txid_behavior(name):
    fixture_path = next(path for path in FIXTURES if path.stem == name)
    fixture = json.loads(fixture_path.read_text())
    raw_tx = bytes.fromhex(fixture['hex'])

    # The final binding signature is authorizing data, so it must not affect
    # ZIP-244's transaction ID.
    changed_signature = raw_tx[:-1] + bytes((raw_tx[-1] ^ 1,))
    changed_txid = DeserializerZcash(changed_signature).read_tx().txid_rev
    assert hash_to_hex_str(changed_txid) == fixture['txid']

    # lockTime is in the effecting header digest and must affect the txid.
    changed_locktime = raw_tx[:12] + bytes((raw_tx[12] ^ 1,)) + raw_tx[13:]
    changed_txid = DeserializerZcash(changed_locktime).read_tx().txid_rev
    assert hash_to_hex_str(changed_txid) != fixture['txid']


def test_zcash_fee_adjustment_is_included_in_mempool_fee():
    pool = object.__new__(MemPool)
    pool.hashXs = defaultdict(set)
    pool.txs = {}
    pool.txo_to_spender = {}
    prevout = (b'P' * 32, 0)
    txid = b'T' * 32
    tx = MemPoolTx(
        prevouts=(prevout,), in_pairs=None, out_pairs=((b'X' * 11, 60),),
        fee=0, size=100, fee_adjustment=7,
    )
    deferred, _ = pool._accept_transactions(
        tx_map={txid: tx}, utxo_map={prevout: (b'X' * 11, 100)},
        touched_hashxs=set(), touched_outpoints=set(), topologically_sort=False,
    )

    assert deferred == {}
    assert pool.txs[txid].fee == 47


def test_zcash_negative_shielded_aware_fee_is_not_silently_clamped():
    pool = object.__new__(MemPool)
    pool.hashXs = defaultdict(set)
    pool.txs = {}
    pool.txo_to_spender = {}
    prevout = (b'P' * 32, 0)
    tx = MemPoolTx(
        prevouts=(prevout,), in_pairs=None, out_pairs=((b'X' * 11, 100),),
        fee=0, size=100, fee_adjustment=-1,
    )

    with pytest.raises(ValueError, match='negative shielded-aware'):
        pool._accept_transactions(
            tx_map={b'T' * 32: tx}, utxo_map={prevout: (b'X' * 11, 100)},
            touched_hashxs=set(), touched_outpoints=set(), topologically_sort=False,
        )

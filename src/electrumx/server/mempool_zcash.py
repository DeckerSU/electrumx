# Copyright (c) 2026, the ElectrumX authors

"""Zcash mempool fee accounting for transactions with shielded bundles."""

import itertools
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Dict, Set

from aiorpcx import run_in_thread

from electrumx.lib.hash import hash_to_hex_str
from electrumx.lib.tx import SkipTxDeserialize, TxOutpoint
from electrumx.server.mempool import DB_UTXO_MAP, MemPool, MemPoolTx


@dataclass(slots=True)
class ZcashMemPoolTx(MemPoolTx):
    # Net public value balance of shielded bundles, in zatoshis.
    fee_adjustment: int = 0


class ZcashMemPool(MemPool):
    """Preserve shielded value balances when calculating Zcash mempool fees.

    These two overrides mirror the base implementation except for retaining
    fee_adjustment and using it in the fee equation.  Keep the shared fetching
    and acceptance behaviour in sync when those base methods change.  The
    duplication leaves MemPool and MemPoolTx unchanged for other coins.
    """

    def _accept_transactions(
            self,
            *,
            tx_map: Dict[bytes, ZcashMemPoolTx],  # txid_rev->tx
            utxo_map: DB_UTXO_MAP,  # prevout->(hashX,value_in_sats)
            touched_hashxs: Set[bytes],  # set of hashXs
            touched_outpoints: Set[TxOutpoint],  # set of outpoints
            topologically_sort: bool,
    ) -> tuple[dict[bytes, ZcashMemPoolTx], DB_UTXO_MAP]:
        '''Accept transactions in tx_map to the mempool if all their inputs
        can be found in the existing mempool or a utxo_map from the
        DB.

        Returns an (unprocessed tx_map, unspent utxo_map) pair.
        '''
        hashXs = self.hashXs
        txs = self.txs
        txo_to_spender = self.txo_to_spender

        if topologically_sort:
            # Sort tx_map so that parent txs come first (in case of unconf chain).
            # This is just an optimization so that fewer txs will get "deferred", leading to better perf.
            # Dependencies outside tx_map (already in chainstate (utxo_map), or our mempool (self.txs),
            # or MISSING) are ignored.
            cand_to_parents = {
                txid: {parent_txid for (parent_txid, txout_idx) in tx.prevouts
                       if parent_txid in tx_map}
                for (txid, tx) in tx_map.items()}
            cand_txids = list(TopologicalSorter(cand_to_parents).static_order())
            assert len(tx_map) == len(cand_to_parents) == len(cand_txids), \
                f"{len(tx_map)=}, {len(cand_to_parents)=}, {len(cand_txids)=}"
            del cand_to_parents
        else:
            cand_txids = tx_map.keys()

        deferred = {}  # type: dict[bytes, ZcashMemPoolTx]
        unspent = set(utxo_map)
        # Try to find all prevouts so we can accept the candidate TXs from tx_map into our mempool
        for txid_rev in cand_txids:
            tx = tx_map[txid_rev]
            in_pairs = []
            try:
                for prevout in tx.prevouts:
                    # first, look for parent tx among confirmed UTXOs:
                    utxo = utxo_map.get(prevout)
                    if not utxo:  # second, look for parent tx in mempool
                        prev_hash, prev_index = prevout
                        # Raises KeyError if prev_hash is not in txs
                        utxo = txs[prev_hash].out_pairs[prev_index]
                    in_pairs.append(utxo)
            except KeyError:
                deferred[txid_rev] = tx
                continue

            # Spend the prevouts
            unspent.difference_update(tx.prevouts)

            # Save the in_pairs, compute the fee and accept the TX.  Zcash
            # shielded bundle value balances are part of the fee equation.
            tx.in_pairs = tuple(in_pairs)
            fee = (sum(v for _, v in tx.in_pairs) -
                   sum(v for _, v in tx.out_pairs) + tx.fee_adjustment)
            if fee < 0 and tx.fee_adjustment:
                raise ValueError(f'negative shielded-aware mempool fee for {txid_rev.hex()}')
            # Preserve the generation-like input handling for existing coins,
            # where inputs can be intentionally absent from the UTXO map.
            tx.fee = max(0, fee)
            txs[txid_rev] = tx

            for hashX, _value in itertools.chain(tx.in_pairs, tx.out_pairs):
                touched_hashxs.add(hashX)
                hashXs[hashX].add(txid_rev)
            for prevout in tx.prevouts:
                txo_to_spender[prevout] = txid_rev
                touched_outpoints.add(prevout)
            for out_idx, out_pair in enumerate(tx.out_pairs):
                touched_outpoints.add((txid_rev, out_idx))

        return deferred, {prevout: utxo_map[prevout] for prevout in unspent}

    async def _fetch_raw_txs_and_utxos(
            self,
            *,
            new_txids_rev: set[bytes],  # (some) new candidate txs for our mempool
            all_txids_rev: set[bytes],  # complete view of daemon's mempool
    ) -> tuple[dict[bytes, ZcashMemPoolTx], DB_UTXO_MAP]:
        '''Fetch a list of mempool transactions, and lookup corresponding UTXOs in the DB.'''
        txids_hum_iter = (hash_to_hex_str(hash) for hash in new_txids_rev)
        raw_txs = await self.api.raw_transactions(txids_hum_iter)

        def deserialize_txs() -> Dict[bytes, ZcashMemPoolTx]:
            """This function is pure"""
            to_hashX = self.coin.hashX_from_script
            deserializer = self.coin.DESERIALIZER

            txs = {}  # type: Dict[bytes, ZcashMemPoolTx]
            for txid_rev, raw_tx in zip(new_txids_rev, raw_txs):
                # The daemon may have evicted the tx from its
                # mempool or it may have gotten in a block
                if not raw_tx:
                    continue
                try:
                    tx, tx_size = deserializer(raw_tx).read_tx_and_vsize()
                except SkipTxDeserialize as ex:
                    self.logger.debug(f'skipping tx {hash_to_hex_str(txid_rev)}: {ex}')
                    continue
                # Convert the inputs and outputs into (hashX, value) pairs
                # Drop generation-like inputs from ZcashMemPoolTx.prevouts
                txin_pairs = tuple((txin.prev_txid_rev, txin.prev_idx)
                                   for txin in tx.inputs
                                   if not txin.is_generation())
                txout_pairs = tuple((to_hashX(txout.pk_script), txout.value)
                                    for txout in tx.outputs)
                txs[txid_rev] = ZcashMemPoolTx(
                    prevouts=txin_pairs,
                    in_pairs=None,
                    out_pairs=txout_pairs,
                    fee=0,
                    size=tx_size,
                    fee_adjustment=getattr(tx, 'fee_adjustment', 0),
                )
            return txs

        # Thread this potentially slow operation so as not to block
        tx_map = await run_in_thread(deserialize_txs)  # type: Dict[bytes, ZcashMemPoolTx]

        # Determine all prevouts not in the mempool, and fetch the
        # UTXO information from the database.  Failed prevout lookups
        # return None - concurrent database updates happen - which is
        # relied upon by _accept_transactions. Ignore prevouts that are
        # generation-like.
        prevouts = tuple(prevout for tx in tx_map.values()
                         for prevout in tx.prevouts
                         if prevout[0] not in all_txids_rev)
        utxos = await self.api.lookup_utxos(prevouts)
        utxo_map = {prevout: utxo for prevout, utxo in zip(prevouts, utxos)}

        return tx_map, utxo_map

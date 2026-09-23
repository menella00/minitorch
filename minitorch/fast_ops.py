from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Type

import numpy as np
from numba import njit, prange
from numba.core.registry import CPUDispatcher

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.


def _ensure_njit(fn: Any) -> Any:
    if isinstance(fn, CPUDispatcher):
        return fn
    return njit()(fn)


# JIT compile core indexing functions without aggressive inlining
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        f = tensor_map(_ensure_njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_zip(_ensure_njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(_ensure_njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """

        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    f = _ensure_njit(fn)

    def _map(
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        if (
            len(out_shape) == len(in_shape)
            and np.array_equal(out_shape, in_shape)
            and np.array_equal(out_strides, in_strides)
        ):
            for i in prange(len(out_storage)):
                out_storage[i] = f(in_storage[i])
            return

        out_size = int(np.prod(out_shape))
        for ordinal in prange(out_size):
            out_index = np.empty(len(out_shape), dtype=np.int32)
            in_index = np.empty(len(in_shape), dtype=np.int32)
            to_index(ordinal, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)
            out_storage[out_pos] = f(in_storage[in_pos])

    return njit(parallel=True)(_map)


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    f = _ensure_njit(fn)

    def _zip(
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        if (
            len(out_shape) == len(a_shape)
            and len(out_shape) == len(b_shape)
            and np.array_equal(out_shape, a_shape)
            and np.array_equal(out_shape, b_shape)
            and np.array_equal(out_strides, a_strides)
            and np.array_equal(out_strides, b_strides)
        ):
            for i in prange(len(out_storage)):
                out_storage[i] = f(a_storage[i], b_storage[i])
            return

        out_size = int(np.prod(out_shape))
        for ordinal in prange(out_size):
            out_index = np.empty(len(out_shape), dtype=np.int32)
            a_index = np.empty(len(a_shape), dtype=np.int32)
            b_index = np.empty(len(b_shape), dtype=np.int32)

            to_index(ordinal, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            out_storage[out_pos] = f(a_storage[a_pos], b_storage[b_pos])

    return njit(parallel=True)(_zip)


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    f = _ensure_njit(fn)

    def _reduce(
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        out_size = int(np.prod(out_shape))
        reduce_size = a_shape[reduce_dim]
        reduce_stride = a_strides[reduce_dim]

        for ordinal in prange(out_size):
            out_index = np.empty(len(out_shape), dtype=np.int32)
            to_index(ordinal, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)

            a_start_pos = index_to_position(out_index, a_strides)

            acc = out_storage[out_pos]
            for step in range(reduce_size):
                acc = f(acc, a_storage[a_start_pos + step * reduce_stride])
            out_storage[out_pos] = acc

    return njit(parallel=True)(_reduce)


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    NUMBA tensor matrix multiply function.
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    batches = out_shape[0]
    m_dim = out_shape[1]
    n_dim = out_shape[2]
    k_dim = a_shape[2]

    for n in prange(batches):
        for i in range(m_dim):
            for j in range(n_dim):
                out_pos = (
                    n * out_strides[0]
                    + i * out_strides[1]
                    + j * out_strides[2]
                )
                a_start_pos = (
                    n * a_batch_stride
                    + i * a_strides[1]
                )
                b_start_pos = (
                    n * b_batch_stride
                    + j * b_strides[2]
                )

                acc = 0.0
                for k in range(k_dim):
                    acc += (
                        a_storage[a_start_pos + k * a_strides[2]]
                        * b_storage[b_start_pos + k * b_strides[1]]
                    )
                out[out_pos] = acc


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
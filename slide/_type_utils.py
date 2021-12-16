from typing import Dict, Iterable, Tuple
import pyarrow as pa
from triad.utils.pyarrow import TRIAD_DEFAULT_TIMESTAMP

_ORDERED_PYARROW_TYPES = [
    pa.bool_(),
    pa.int8(),
    pa.uint8(),
    pa.int16(),
    pa.uint16(),
    pa.int32(),
    pa.uint32(),
    pa.int64(),
    pa.uint64(),
    pa.float16(),
    pa.float32(),
    pa.float64(),
    pa.string(),
]


def _generate_union_inference_types() -> Iterable[  # noqa: C901
    Tuple[pa.DataType, pa.DataType, pa.DataType]
]:
    a = pa.bool_()
    for b in _ORDERED_PYARROW_TYPES[1:]:
        yield a, b, b
    a = pa.int8()
    yield a, pa.bool_(), a
    yield a, pa.uint8(), pa.int16()
    yield a, pa.uint16(), pa.int32()
    yield a, pa.uint32(), pa.int64()
    yield a, pa.uint64(), pa.float64()
    for b in [
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.float16(),
        pa.float32(),
        pa.float64(),
        pa.string(),
    ]:
        yield a, b, b

    a = pa.uint8()
    yield a, pa.bool_(), a
    yield a, pa.int8(), pa.int16()
    for b in _ORDERED_PYARROW_TYPES[3:]:
        yield a, b, b

    a = pa.int16()
    for b in _ORDERED_PYARROW_TYPES[:3]:
        yield a, b, a
    yield a, pa.uint16(), pa.int32()
    yield a, pa.uint32(), pa.int64()
    yield a, pa.uint64(), pa.float64()
    for b in [
        pa.int32(),
        pa.int64(),
        pa.float16(),
        pa.float32(),
        pa.float64(),
        pa.string(),
    ]:
        yield a, b, b

    a = pa.uint16()
    yield a, pa.bool_(), a
    yield a, pa.int8(), pa.int32()
    yield a, pa.uint8(), a
    yield a, pa.int16(), pa.int32()
    for b in _ORDERED_PYARROW_TYPES[5:]:
        yield a, b, b

    a = pa.int32()
    for b in _ORDERED_PYARROW_TYPES[:5]:
        yield a, b, a
    yield a, pa.uint32(), pa.int64()
    yield a, pa.uint64(), pa.float64()
    for b in [
        pa.int64(),
        pa.float16(),
        pa.float32(),
        pa.float64(),
        pa.string(),
    ]:
        yield a, b, b

    a = pa.uint32()
    yield a, pa.bool_(), a
    yield a, pa.int8(), pa.int64()
    yield a, pa.uint8(), a
    yield a, pa.int16(), pa.int64()
    yield a, pa.uint16(), a
    for b in _ORDERED_PYARROW_TYPES[7:]:
        yield a, b, b

    a = pa.int64()
    for b in _ORDERED_PYARROW_TYPES[:7]:
        yield a, b, a
    yield a, pa.uint64(), pa.float64()
    for b in [
        pa.float16(),
        pa.float32(),
        pa.float64(),
        pa.string(),
    ]:
        yield a, b, b

    a = pa.uint64()
    yield a, pa.bool_(), a
    yield a, pa.int8(), pa.float64()
    yield a, pa.uint8(), a
    yield a, pa.int16(), pa.float64()
    yield a, pa.uint16(), a
    yield a, pa.int32(), pa.float64()
    yield a, pa.uint32(), a
    for b in _ORDERED_PYARROW_TYPES[9:]:
        yield a, b, b

    a = pa.float16()
    for b in _ORDERED_PYARROW_TYPES[:9]:
        yield a, b, a
    for b in _ORDERED_PYARROW_TYPES[10:]:
        yield a, b, b

    a = pa.float32()
    for b in _ORDERED_PYARROW_TYPES[:10]:
        yield a, b, a
    for b in _ORDERED_PYARROW_TYPES[11:]:
        yield a, b, b

    a = pa.float64()
    for b in _ORDERED_PYARROW_TYPES[:11]:
        yield a, b, a
    for b in _ORDERED_PYARROW_TYPES[10:]:
        yield a, b, b

    a = pa.string()
    for b in _ORDERED_PYARROW_TYPES[:12]:
        yield a, b, a

    yield pa.date32(), pa.date64(), pa.date64()
    yield pa.date64(), pa.date32(), pa.date64()


_UNION_INFERENCE_DICT: Dict[Tuple[pa.DataType, pa.DataType], pa.DataType] = {
    (x[0], x[1]): x[2] for x in _generate_union_inference_types()
}


def infer_union_type(  # noqa: C901
    t1: pa.DataType,
    t2: pa.DataType,
) -> pa.DataType:
    if t1 == t2:
        return t1
    if pa.types.is_timestamp(t1):
        if pa.types.is_timestamp(t2) or pa.types.is_date(t2):
            return TRIAD_DEFAULT_TIMESTAMP
        elif pa.types.is_string(t2):
            return pa.string()
        raise ValueError(f"can't infer unioned schema for {t1} and {t2}")
    if pa.types.is_timestamp(t2):
        if pa.types.is_timestamp(t1) or pa.types.is_date(t1):
            return TRIAD_DEFAULT_TIMESTAMP
        elif pa.types.is_string(t1):
            return pa.string()
        raise ValueError(f"can't infer unioned schema for {t1} and {t2}")
    if pa.types.is_nested(t1) or pa.types.is_nested(t2):
        raise ValueError(f"can't infer unioned schema for {t1} and {t2}")
    if pa.types.is_binary(t1) or pa.types.is_binary(t2):
        raise ValueError(f"can't infer unioned schema for {t1} and {t2}")
    key = (t1, t2)
    if key in _UNION_INFERENCE_DICT:
        return _UNION_INFERENCE_DICT[key]
    raise ValueError(  # pragma: no cover
        f"can't infer unioned schema for {t1} and {t2}"
    )

import pyarrow as pa
from pytest import raises
from triad import Schema

from slide._type_utils import infer_union_type


def test_infer_union_type():
    schema = Schema(
        "a:int32,b:float32,c:string,d:datetime,e:date,f:[int],g:{a:str},h:binary"
    )
    assert pa.int32() == infer_union_type(schema["a"].type, schema["a"].type)
    assert pa.float32() == infer_union_type(schema["a"].type, schema["b"].type)
    assert pa.string() == infer_union_type(schema["a"].type, schema["c"].type)
    assert pa.string() == infer_union_type(schema["c"].type, schema["a"].type)
    assert schema["d"].type == infer_union_type(schema["d"].type, schema["d"].type)
    assert schema["d"].type == infer_union_type(schema["d"].type, schema["e"].type)
    assert schema["d"].type == infer_union_type(schema["e"].type, schema["d"].type)
    assert pa.string() == infer_union_type(schema["d"].type, schema["c"].type)
    assert pa.string() == infer_union_type(schema["c"].type, schema["d"].type)
    assert schema["f"].type == infer_union_type(schema["f"].type, schema["f"].type)
    assert schema["g"].type == infer_union_type(schema["g"].type, schema["g"].type)

    raises(ValueError, lambda: infer_union_type(schema["f"].type, schema["g"].type))
    raises(ValueError, lambda: infer_union_type(schema["c"].type, schema["g"].type))
    raises(ValueError, lambda: infer_union_type(schema["d"].type, schema["a"].type))
    raises(ValueError, lambda: infer_union_type(schema["a"].type, schema["d"].type))
    raises(ValueError, lambda: infer_union_type(schema["a"].type, schema["h"].type))
    raises(ValueError, lambda: infer_union_type(schema["h"].type, schema["a"].type))

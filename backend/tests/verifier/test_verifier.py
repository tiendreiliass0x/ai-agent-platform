import pytest

from app.verifier import Verifier, VerifierError
from app.tooling.models import OperationSpec, SideEffect


def make_operation(preconditions=None, postconditions=None):
    return OperationSpec(
        op_id="op",
        method="POST",
        path="/",
        side_effect=SideEffect.WRITE,
        description="",
        args_schema={},
        returns={},
        preconditions=preconditions or [],
        postconditions=postconditions or [],
    )


def test_preconditions_pass():
    verifier = Verifier()
    op = make_operation(preconditions=["args['value'] > 0"])
    verifier.verify_preconditions(op, {"value": 5})


def test_preconditions_fail():
    verifier = Verifier()
    op = make_operation(preconditions=["args.value > 0"])
    with pytest.raises(VerifierError):
        verifier.verify_preconditions(op, {"value": -1})


def test_postconditions_pass():
    verifier = Verifier()
    op = make_operation(postconditions=["result['status'] == 'ok'"])
    verifier.verify_postconditions(op, args={}, result={"status": "ok"})


def test_postconditions_fail():
    verifier = Verifier()
    op = make_operation(postconditions=["result['count'] == args['expected']"])
    with pytest.raises(VerifierError):
        verifier.verify_postconditions(op, args={"expected": 2}, result={"count": 1})


def test_disallowed_expression():
    verifier = Verifier()
    op = make_operation(preconditions=["__import__('os').system('ls')"])
    with pytest.raises(VerifierError):
        verifier.verify_preconditions(op, {})

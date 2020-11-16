import traceback

from factor import Factor
from factor_utils import factor_evidence, factor_marginalize, factor_product

STR_OUTPUT_MISMATCH = "Output does not match."


def wrap_test(func):
    def inner():
        func_name = func.__name__.replace("test_", "")
        try:
            func()
            print("{}: PASSED".format(func_name))
        except Exception as e:
            print("{}: FAILED, reason: {} ***".format(func_name, str(e)))
            print(traceback.format_exc())

    return inner


@wrap_test
def test_factor_product():
    # Test case 1
    # factorA contains P(X_0)
    factor0 = Factor(var=[0], card=[2], val=[0.8, 0.2])
    # factor1 contains P(X_1|X_0)
    factor1 = Factor(var=[0, 1], card=[2, 2], val=[0.4, 0.55, 0.6, 0.45])

    correct = Factor(var=[0, 1], card=[2, 2], val=[0.32, 0.11, 0.48, 0.09])

    output = factor_product(factor0, factor1)
    assert output == correct, f"test_factor_product_1 {STR_OUTPUT_MISMATCH}"

    # Test case 2: No variables in common
    # factorA contains P(X_0)
    factor0 = Factor(var=[0], card=[2], val=[0.8, 0.2])
    # factor1 contains P(X_1)
    factor1 = Factor(var=[1], card=[2], val=[0.4, 0.6])

    correct = Factor(var=[0, 1], card=[2, 2], val=[0.32, 0.08, 0.48, 0.12])

    output = factor_product(factor0, factor1)
    assert output == correct, f"test_factor_product_2 {STR_OUTPUT_MISMATCH}"


@wrap_test
def test_factor_marginalize():
    factor = Factor(var=[2, 3], card=[2, 3], val=[0.4, 0.35, 0.6, 0.45, 0.0, 0.2])
    vars_to_marginalize_out = [2]

    correct = Factor(var=[3], card=[3], val=[0.75, 1.05, 0.2])

    output = factor_marginalize(factor, vars_to_marginalize_out)
    assert output == correct, STR_OUTPUT_MISMATCH


@wrap_test
def test_factor_evidence():
    # Single evidence
    factor = Factor(var=[0, 1], card=[2, 3], val=[0.4, 0.35, 0.6, 0.45, 0.0, 0.2])
    evidence = {0: 1}

    correct = Factor(var=[1], card=[3], val=[0.35, 0.45, 0.2])

    output = factor_evidence(factor, evidence)
    assert output == correct, STR_OUTPUT_MISMATCH

    # Multiple evidences
    factor = Factor(
        var=[0, 1, 2], card=[2, 2, 2], val=[0.4, 0.35, 0.6, 0.45, 0.0, 0.2, 0.5, 0.7]
    )
    evidence = {0: 1, 1: 0}

    correct = Factor(var=[2], card=[2], val=[0.35, 0.2])

    output = factor_evidence(factor, evidence)
    assert output == correct, STR_OUTPUT_MISMATCH


if __name__ == "__main__":
    test_factor_product()
    test_factor_marginalize()
    test_factor_evidence()

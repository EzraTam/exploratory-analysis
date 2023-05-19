from typing import List
import pandas as pd
from scipy import stats

basic_text = (
    "The p-value of the __TEST__ test is __RESULT_COMP__ than 0.05.\n"
    "Therefore, we __IMPLICATION__1 the null hypothesis and infer according to this test, that __IMPLICATION__2"
)


def _text_util(
    test_display_nm: str,
    result_comp: str,
    implication_1: str,
    implication_2: str,
    p_value,
):
    _text = basic_text.replace("__TEST__", test_display_nm)
    _text = (
        _text.replace("__RESULT_COMP__", result_comp)
        .replace("__IMPLICATION__1", implication_1)
        .replace("__IMPLICATION__2", implication_2)
    )

    print("\033[1m" + f"Result of the {test_display_nm} Test:" + "\033[0m")
    print(f"The p-value of {test_display_nm} test is {round(p_value,2)}")
    print(_text)


def test_normality(data: pd.Series) -> None:
    tests = {
        "shapiro": {"method": stats.shapiro, "display_nm": "Shapiro"},
        "ks": {
            "method": lambda data: stats.kstest(
                (data - data.mean()) / data.std(), stats.norm.cdf
            ),
            "display_nm": "Kolmogorov-Smirnov",
        },
    }

    for _test in ["shapiro", "ks"]:
        _test_display_nm = tests[_test]["display_nm"]
        _test = tests[_test]["method"](data)
        _result_comp = "greater" if _test.pvalue >= 0.05 else "less"
        _implication_1 = "accept" if _test.pvalue >= 0.05 else "reject"
        _implication_2 = (
            "the observation is normally distributed"
            if _test.pvalue >= 0.05
            else "the observation is not normally distributed"
        )
        _text_util(
            test_display_nm=_test_display_nm,
            result_comp=_result_comp,
            implication_1=_implication_1,
            implication_2=_implication_2,
            p_value=_test.pvalue,
        )


def test_equal_distributions(data: List[pd.Series]) -> None:
    _test_display_nm = "Kolmogorov-Smirnov"
    _test = stats.kstest(*data)
    _result_comp = "greater" if _test.pvalue >= 0.05 else "less"
    _implication_1 = "accept" if _test.pvalue >= 0.05 else "reject"
    _implication_2 = (
        "the observations have the same distribution"
        if _test.pvalue >= 0.05
        else "the observations have not the same distribution"
    )
    _text_util(
            test_display_nm=_test_display_nm,
            result_comp=_result_comp,
            implication_1=_implication_1,
            implication_2=_implication_2,
            p_value=_test.pvalue,
        )


def test_equal_variances(data: List[pd.Series]) -> None:
    _test_display_nm = "Levene"
    _test = stats.levene(*data)
    _result_comp = "greater" if _test.pvalue >= 0.05 else "less"
    _implication_1 = "accept" if _test.pvalue >= 0.05 else "reject"
    _implication_2 = (
        "the observations have the same variances"
        if _test.pvalue >= 0.05
        else "the observations have not the same variances"
    )
    _text_util(
            test_display_nm=_test_display_nm,
            result_comp=_result_comp,
            implication_1=_implication_1,
            implication_2=_implication_2,
            p_value=_test.pvalue,
        )
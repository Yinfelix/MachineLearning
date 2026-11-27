from src import config as C


def test_scheme_sizes():
    expected = {"A": 51, "B": 45, "C": 23, "D": 28, "S": 39, "U": 19, "T": 38, "V": 18}
    for scheme, n in expected.items():
        numeric, categorical = C.scheme_columns(scheme)
        assert len(numeric) + len(categorical) == n


def test_schemes_are_nested_as_intended():
    A = set(C.SCHEMES["A"]["numeric"]) | set(C.SCHEMES["A"]["categorical"])
    B = set(C.SCHEMES["B"]["numeric"]) | set(C.SCHEMES["B"]["categorical"])
    S = set(C.SCHEMES["S"]["numeric"]) | set(C.SCHEMES["S"]["categorical"])
    T = set(C.SCHEMES["T"]["numeric"]) | set(C.SCHEMES["T"]["categorical"])
    C_ = set(C.SCHEMES["C"]["numeric"]) | set(C.SCHEMES["C"]["categorical"])
    D = set(C.SCHEMES["D"]["numeric"]) | set(C.SCHEMES["D"]["categorical"])
    U = set(C.SCHEMES["U"]["numeric"]) | set(C.SCHEMES["U"]["categorical"])
    V = set(C.SCHEMES["V"]["numeric"]) | set(C.SCHEMES["V"]["categorical"])
    assert B < A and S < B and T < S and U < C_ and V < U
    assert C_ | D == A
    assert C_ & D == set()


def test_shortcuts_removed_from_strict_schemes():
    for s in C.SHORTCUT_FEATURES:
        assert s not in C.SCHEMES["B"]["numeric"]
        assert s not in C.SCHEMES["U"]["numeric"]
        assert s not in C.SCHEMES["S"]["numeric"]
        assert s not in C.SCHEMES["T"]["numeric"]
    for s in C.PAGE_SIZE_SHORTCUTS:
        assert s not in C.SCHEMES["S"]["numeric"]
        assert s not in C.SCHEMES["T"]["numeric"]
    assert "IsHTTPS" not in C.SCHEMES["T"]["numeric"]
    assert "IsHTTPS" not in C.SCHEMES["V"]["numeric"]

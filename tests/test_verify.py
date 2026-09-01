"""Unit tests for the verification rules. These run without Tesseract."""
from app.verify import (GOVERNMENT_WARNING, Application, Status, check_abv, check_government_warning,
                        check_net_contents, check_text_field, parse_abv, parse_volume_ml, verify)

LABEL = f"""OLD TOM DISTILLERY
Kentucky Straight Bourbon Whiskey
45% Alc./Vol. (90 Proof)
750 mL
Distilled and Bottled by Old Tom Distillery, Bardstown, KY
{GOVERNMENT_WARNING}"""


# ---- brand / text fields --------------------------------------------------
def test_brand_exact():
    r = check_text_field("brand_name", "Brand", "OLD TOM DISTILLERY", LABEL)
    assert r.status == Status.PASS and r.message == "Exact match."


def test_brand_case_difference_is_pass_with_note():
    # Dave's case: STONE'S THROW on label, Stone's Throw in application.
    r = check_text_field("brand_name", "Brand", "Stone's Throw", "STONE'S THROW\nAmerican Single Malt")
    assert r.status == Status.PASS
    assert "capitalisation" in r.message


def test_brand_curly_apostrophe_and_missing_punctuation():
    r = check_text_field("brand_name", "Brand", "Stone’s Throw", "STONES THROW")
    assert r.status == Status.WARN          # matches ignoring punctuation -> human confirms


def test_brand_ocr_noise_is_warn():
    r = check_text_field("brand_name", "Brand", "OLD TOM DISTILLERY", "OLD T0M DISTILLERY\n45%")
    assert r.status == Status.WARN


def test_brand_missing_is_fail():
    r = check_text_field("brand_name", "Brand", "HARBOR LIGHT GIN", LABEL)
    assert r.status == Status.FAIL


def test_empty_field_is_skip():
    assert check_text_field("brand_name", "Brand", "", LABEL).status == Status.SKIP


# ---- ABV ------------------------------------------------------------------
def test_parse_abv_variants():
    assert parse_abv("45% Alc./Vol. (90 Proof)") == 45.0
    assert parse_abv("45") == 45.0
    assert parse_abv("12,5 %") == 12.5
    assert parse_abv("ninety proof") is None


def test_abv_match_and_proof_consistency():
    assert check_abv("45%", LABEL).status == Status.PASS
    r = check_abv("45%", "45% Alc./Vol. (80 Proof)")
    assert r.status == Status.WARN and "proof" in r.message.lower()


def test_abv_mismatch():
    assert check_abv("40%", LABEL).status == Status.FAIL


def test_abv_close_miss_is_warn():
    assert check_abv("45.5%", LABEL).status == Status.WARN


def test_abv_missing():
    assert check_abv("45%", "OLD TOM\n750 mL").status == Status.FAIL


def test_abv_prefers_percentage_near_alc_vol():
    text = "Made with 100% corn\n45% Alc./Vol."
    assert check_abv("45%", text).status == Status.PASS


# ---- net contents ---------------------------------------------------------
def test_volume_units():
    assert parse_volume_ml("750 mL") == 750
    assert parse_volume_ml("1 L") == 1000
    assert parse_volume_ml("75 cl") == 750
    assert abs(parse_volume_ml("25.4 fl. oz.") - 751.2) < 1


def test_net_contents_cross_unit_match():
    assert check_net_contents("750 mL", "Contents: 75 cl").status == Status.PASS
    assert check_net_contents("750 mL", "1 L").status == Status.FAIL
    assert check_net_contents("750 mL", "no volume here").status == Status.FAIL


# ---- government warning ---------------------------------------------------
def test_warning_exact():
    r, diffs = check_government_warning(LABEL)
    assert r.status == Status.PASS and diffs == []


def test_warning_line_wrapped_and_comma_dropped_by_ocr():
    wrapped = GOVERNMENT_WARNING.replace("General, women", "General\nwomen").replace("car or", "car or\n")
    r, _ = check_government_warning(wrapped)
    assert r.status == Status.PASS
    assert "Punctuation" in r.message


def test_warning_title_case_header_fails():
    # Jenny's case.
    r, _ = check_government_warning(GOVERNMENT_WARNING.replace("GOVERNMENT WARNING:", "Government Warning:"))
    assert r.status == Status.FAIL and "capital letters" in r.message


def test_warning_altered_wording_fails_with_diff():
    altered = GOVERNMENT_WARNING.replace("should not drink", "should avoid drinking")
    r, diffs = check_government_warning(altered)
    assert r.status == Status.FAIL
    assert any("not drink" in d["expected"] for d in diffs)


def test_warning_single_ocr_typo_is_warn():
    r, diffs = check_government_warning(GOVERNMENT_WARNING.replace("machinery", "machinory"))
    assert r.status == Status.WARN and diffs


def test_warning_missing():
    r, _ = check_government_warning("OLD TOM\n45%")
    assert r.status == Status.FAIL


# ---- overall --------------------------------------------------------------
def test_overall_rollup():
    app = Application(brand_name="OLD TOM DISTILLERY", class_type="Kentucky Straight Bourbon Whiskey", abv="45%", net_contents="750 mL")
    assert verify(app, LABEL)["overall"] == "pass"
    assert verify(Application(brand_name="OLD TOM DISTILLERY", abv="40%"), LABEL)["overall"] == "fail"
    assert verify(Application(brand_name="OLD T0M DISTILLERY"), LABEL)["overall"] == "warn"

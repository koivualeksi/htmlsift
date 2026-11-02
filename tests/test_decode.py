"""decode_html: bytes -> str."""
import pytest

from htmlsift._decode import decode_html, prepare_input

FI = "<p>Päivää “hei”</p>"
RU = "<p>Привет, мир</p>"
R = "\ufffd"


def meta(charset):
    return f'<meta charset="{charset}">'


def test_utf8_bytes_decode():
    assert decode_html("<p>Päivää – 日本語</p>".encode()) == "<p>Päivää – 日本語</p>"


def test_empty_bytes():
    assert decode_html(b"") == ""


@pytest.mark.parametrize("bad", [None, "<p>x</p>", bytearray(b"x"), memoryview(b"x"), 42])
def test_non_bytes_rejected(bad):
    with pytest.raises(TypeError):
        decode_html(bad)


# BOM

@pytest.mark.parametrize("bom, codec", [
    (b"\xef\xbb\xbf", "utf-8"), (b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be")])
def test_bom_decoded_and_stripped(bom, codec):
    assert decode_html(bom + FI.encode(codec)) == FI


def test_bom_beats_encoding_arg():
    assert decode_html(b"\xef\xbb\xbf" + FI.encode(), encoding="windows-1252") == FI


def test_bom_beats_meta():
    assert decode_html(b"\xef\xbb\xbf" + (meta("shift_jis") + FI).encode()).endswith(FI)


def test_bom_is_final():
    assert decode_html(b"\xef\xbb\xbf" + FI.encode() + b"\xe4") == FI + R


def test_bom_only():
    assert decode_html(b"\xef\xbb\xbf") == ""


# encoding= argument

def test_encoding_arg():
    assert decode_html(RU.encode("cp1251"), encoding="windows-1251") == RU


def test_encoding_arg_beats_wrong_meta():
    html = meta("shift_jis") + FI
    assert decode_html(html.encode(), encoding="utf-8") == html


def test_encoding_arg_whitespace_stripped():
    assert decode_html(RU.encode("cp1251"), encoding=" windows-1251 ") == RU


def test_encoding_arg_latin1_is_cp1252():
    data = FI.encode()
    assert decode_html(data, encoding="latin1") == data.decode("cp1252", errors="replace")


def test_unknown_encoding_arg_falls_through_to_meta():
    html = meta("windows-1251") + RU
    assert decode_html(html.encode("cp1251"), encoding="bogus") == html


def test_declared_utf8_is_trusted():
    assert decode_html(FI.encode("cp1252"), encoding="utf-8") == f"<p>P{R}iv{R}{R} {R}hei{R}</p>"


# <meta> declaration

def test_meta_cp1251():
    html = meta("windows-1251") + RU
    assert decode_html(html.encode("cp1251")) == html


def test_meta_iso_8859_1_is_cp1252():
    data = (meta("iso-8859-1") + FI).encode()
    assert decode_html(data) == data.decode("cp1252", errors="replace")


def test_meta_http_equiv_content_first():
    html = ('<meta content="text/html; charset=windows-1251" http-equiv="Content-Type">'
            + RU)
    assert decode_html(html.encode("cp1251")) == html


def test_meta_shift_jis_windows_chars():
    html = meta("shift_jis") + "<p>①テスト</p>"
    assert decode_html(html.encode("cp932")) == html


def test_meta_gb2312_gbk_chars():
    html = meta("gb2312") + "<p>镕</p>"
    assert decode_html(html.encode("gbk")) == html


def test_meta_utf16_read_as_utf8():
    html = meta("utf-16") + FI
    assert decode_html(html.encode()) == html


def test_unknown_meta_falls_through():
    html = meta("bogus") + FI
    assert decode_html(html.encode("cp1252")) == html


def test_declared_utf8_meta_is_trusted():
    html = meta("utf-8") + FI
    assert decode_html(html.encode("cp1252")) == meta("utf-8") + f"<p>P{R}iv{R}{R} {R}hei{R}</p>"


# undeclared

def test_undeclared_cp1252():
    assert decode_html(FI.encode("cp1252")) == FI


def test_cp1252_undefined_byte_replaced():
    assert decode_html(b"<p>a\x81b</p>", encoding="windows-1252") == f"<p>a{R}b</p>"


# prepare_input

def test_prepare_str_unchanged():
    assert prepare_input("<p>x</p>") == "<p>x</p>"


def test_prepare_bytes_reencoded_utf8():
    assert prepare_input(FI.encode("cp1252")) == FI.encode()


def test_prepare_str_with_encoding_rejected():
    with pytest.raises(TypeError):
        prepare_input("<p>x</p>", encoding="utf-8")


@pytest.mark.parametrize("bad", [None, bytearray(b"x"), memoryview(b"x"), 42])
def test_prepare_other_types_rejected(bad):
    with pytest.raises(TypeError):
        prepare_input(bad)

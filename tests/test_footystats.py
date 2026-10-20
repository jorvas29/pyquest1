from futquant.footystats import parse_xg_table

TOTALS = """<html><body>
<table><tr><th>Menu</th></tr><tr><td>a</td></tr></table>
<table class="full-league-table">
 <tr><th>#</th><th>Team</th><th>MP</th><th>xG</th><th>xGA</th><th>GF</th><th>GA</th></tr>
 <tr><td>1</td><td><a href="#">Liverpool FC</a></td><td>2</td><td>5.74</td><td>3.22</td><td>8</td><td>4</td></tr>
 <tr><td>2</td><td>Villarreal CF</td><td>2</td><td>4.36</td><td>3,64</td><td>5</td><td>4</td></tr>
</table></body></html>"""

PER_MATCH = """<table>
 <tr><th>Equipo</th><th>xG / Match</th><th>xGA / Match</th></tr>
 <tr><td>Fenerbahce</td><td>1.62</td><td>2.10</td></tr>
 <tr><td>Slavia Praha</td><td>1.40</td><td>1.75</td></tr>
 <tr><td>Sabah</td><td>0.98</td><td>2.40</td></tr>
</table>"""


def test_totals_are_divided_by_matches():
    rows = parse_xg_table(TOTALS)
    assert [r["equipo"] for r in rows] == ["Liverpool FC", "Villarreal CF"]
    assert rows[0]["xg_per_match"] == 2.87 and rows[1]["xga_per_match"] == 1.82


def test_per_match_headers():
    rows = parse_xg_table(PER_MATCH)
    assert len(rows) == 3
    assert rows[0] == {"equipo": "Fenerbahce", "xg_per_match": 1.62, "xga_per_match": 2.10}


def test_no_xg_table():
    assert parse_xg_table("<table><tr><th>a</th></tr><tr><td>1</td></tr><tr><td>2</td></tr></table>") == []

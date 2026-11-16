# Cluster und gemischte Modelle – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-cluster-gemischt-demo.streamlit.app/)** -->

Eine Demo zu **Regression und verallgemeinerten linearen Modellen** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Leitfrage dieser Demos ist: *Was darf man aus einer angepassten Gleichung schließen, und wann trügt sie?* Hier fällt eine stillschweigende Annahme: **jede Zeile der Tabelle ist eine unabhängige Ziehung**. Die Demos zu Kalibrierung und Schwelle ([`kalibrierung-schwelle-demo`](https://github.com/sebastian-hanisch/kalibrierung-schwelle-demo)) und zur Streuung ([`streuung-wls-demo`](https://github.com/sebastian-hanisch/streuung-wls-demo)) rechnen mit unabhängigen Zeilen; dass die erste sie voraussetzt, ist eine Lesart (ihre Beschreibung nennt es nicht ausdrücklich), keine zitierte Aussage. Ob Kalibrierung und Schwelle unter Clustern leiden, ist hier **nicht gemessen**.

Das Vehikel sind die **Import-Container** der Verweilzeit-Welt ([`verweilzeit-demo`](https://github.com/sebastian-hanisch/verweilzeit-demo)) mit ihren **Kunden als Cluster**: Container desselben Kunden hängen zusammen (gemeinsamer Kundeneffekt auf der log-Verweilzeit). Die Demo vergleicht an einem erfundenen **Kundenmerkmal** („Rahmenvertrag“) und an einem Merkmal **innerhalb** der Kunden („Zollhalt“) das naive Intervall einer Regression, die **cluster-robuste Varianz (CR1)**, das **gemischte Modell (ML und REML)** und die **Kunden-Mittel-Methode**; dazu den **Designeffekt**, den **Rand** σ̂_u² = 0 und die Genauigkeit der Kundenstreuung. Nachbarn: [`ols-projektion-demo`](https://github.com/sebastian-hanisch/ols-projektion-demo), [`hebel-diagnose-demo`](https://github.com/sebastian-hanisch/hebel-diagnose-demo) (der Kunde mit wenigen Containern oder das Kundenmerkmal, das nur wenige Kunden haben, ist ein Hebel auf Kundenebene), [`faktoren-codierung-demo`](https://github.com/sebastian-hanisch/faktoren-codierung-demo), [`zaehldaten-demo`](https://github.com/sebastian-hanisch/zaehldaten-demo) (dort fehlt der Kunde als Merkmal, hier ist er der Cluster), [`modellwahl-demo`](https://github.com/sebastian-hanisch/modellwahl-demo) (die Wahl der Zufallsstruktur ist hier kein Thema). Dasselbe Prinzip „wer ist die Einheit?“ in anderem Material: [`ratio-metric-demo`](https://github.com/sebastian-hanisch/ratio-metric-demo) (Experimente mit Zuteilung je Standort, Verhältniskennzahl; der Standort-Welch dort entspricht der Kunden-Mittel-Methode hier, bei gleich großen Gruppen mit derselben Varianz der Gruppenmittel (Handbeispiel 4); kein neuer Befund), [`synthetic-control-demo`](https://github.com/sebastian-hanisch/synthetic-control-demo) (cluster-robuste Fehler bei **einem** behandelten Cluster), [`aggregationsebene-demo`](https://github.com/sebastian-hanisch/aggregationsebene-demo) und [`uebereinstimmung-demo`](https://github.com/sebastian-hanisch/uebereinstimmung-demo) (Mehrebenenmodelle sind dort nicht gebaut).

## Warum dieses Problem

Eine Tabelle mit 360 Import-Containern von 12 Kunden sieht nach 360 Beobachtungen aus. Für ein Merkmal, das nur der **Kunde** hat, ist sie es nicht: **Container desselben Kunden sind keine unabhängigen Beobachtungen, für den Rahmenvertrag zählt im Wesentlichen die Kundenzahl: von 360 Containern sind etwa 46 wirksam (mehr Container je Kunde erhöhen diese Zahl nur bis zu einer Grenze).** **Wie viele unabhängige Einheiten hat die Tabelle wirklich, welches Intervall darf man glauben, und wann versagt auch das gemischte Modell?**

Im Vehikel (Import-Container, 12 Kunden, im Mittel 30 Container je Kunde, Streuung der Kundeneffekte σ_u = 0,30, also ρ = 0,229; 1000 Historien, Zähleinheit: eine Historie = eine Tabelle aus 12 Kunden; Monte-Carlo-Fehler etwa 0,7 Prozentpunkte bei 95 %) deckt das **naive 95-%-Intervall** des Rahmenvertrags nur **50,8 %**, bei im Mittel 120 Containern je Kunde **26,9 %**: mehr Zeilen machen das Intervall schmaler, nicht richtiger. Cluster-robuste Fehler (CR1) und das gemischte Modell (REML) kommen erst mit dem t(m−1)-Quantil nahe an 95 % (94,5 % und 95,8 %), die Kunden-Mittel-Methode auch (95,7 %). Es kommt aber auch auf die **Verteilung des Merkmals über die Kunden** an: gilt der Rahmenvertrag nur für 2 von 12 Kunden, deckt CR1 mit t(m−1) nur **79,8 %**, REML mit t(m−1) 95,4 % und die Kunden-Mittel-Methode 95,0 %. Und das gemischte Modell liegt bei kleiner Streuung zwischen den Kunden oft am **Rand** σ̂_u² = 0: bei σ_u = 0,10 in 11,1 % der Historien (ML: 19,4 %), und dort deckt REML mit t(m−1) nur 88,3 % gegen 96,7 % sonst.

**Gilt für diese Welt:** Zufallsachsenabschnitt je Kunde, normale Fehler, konstante Streuung (Import: 0,55 auf der log-Skala), Kundeneffekte in jeder Historie frisch gezogen, 6 bis 48 Kunden; ohne Zufallssteigungen und ohne verallgemeinerte gemischte Modelle.

**Der Negativbefund gehört dazu:** für ein Merkmal **innerhalb** der Kunden (Zollhalt) war das naive Intervall in allen 20 vorgerechneten Zeilen **nicht zu schmal** (Überdeckung 94,4 bis 96,4 %); REML mit t(m−1) ist dort sogar zu breit (95,5 bis 99,2 %), weil die Freiheitsgrade für dieses Merkmal nicht m − 1 sind. Ohne Kundeneffekt (σ_u = 0) deckt das naive Intervall 94,9 %: dort ist nichts zu finden.

## Befunde und Korrekturen gegenüber früheren Messungen

Die Zahlen einer früheren Messung (numpy-Zufall, 1000 Historien je Zeile) sind Erwartungen, keine Zahlen dieser Demo. Die Messung der Demo nutzt den ganzzahligen Zufallsgenerator (SplitMix64); jede Zahl dieser README stammt aus ihr. Verschoben oder widerlegt hat sich:

- **REML mit t(m−1) bei 2 von 12 Kunden:** früher 93,0 %, jetzt 95,4 % (Unterschied 2,4 Prozentpunkte, etwa 2,4 Standardfehler der Differenz zweier unabhängiger Messungen); die frühere Zahl wird nicht bestätigt. Dass CR1 hier nur 79,8 % deckt, bleibt (früher dieselbe Zahl).
- **Rand:** früher 52,2 % bei σ_u = 0 und 12,5 % bei σ_u = 0,10 (REML), jetzt 54,6 % und 11,1 %. Die frühere Rechnung hatte eine Lücke (ein Optimum zwischen 0 und dem ersten Gitterpunkt wurde als Rand gemeldet); hier entscheidet die **Ableitung des Kriteriums bei λ = 0** (nicht negativ heißt Rand), ein Regressionstest zeigt den Unterschied (`tests/test_mixed.py`). Im ausgewogenen Fall stimmt der Anteil mit der geschlossenen F-Formel überein (`tests/test_oracle_cluster_gemischt.py`).
- **Naives Intervall bei 120 Containern je Kunde:** früher 29,3 %, jetzt 26,9 % (Unterschied 2,4 Prozentpunkte); die Richtung bleibt.
- **Zollhalt, „Varianz kleiner als naiv (1 − ρ)“:** das gilt nur, wenn der Wert innerhalb jedes Kunden mittelwertfrei ist (Handbeispiel: Varianz 1/8 gegen 1/6 mit Gesamtvarianz, Faktor 3/4). Im Vehikel wechselt der Zollhalt zufällig, dort liegt die wahre Varianz im Median bei 0,99 der erwarteten naiven (12 Kunden, 30 Container je Kunde, 2000 Zuordnungen) und im Median zwischen 0,95 und 1,00 über 36 Einstellungen (Tabelle unten): das naive Intervall ist für den Zollhalt nicht zu schmal.
- **Halbbreite des naiven Intervalls im Handbeispiel:** 3,75 (t(5)·√(32/15)), nicht 3,76 (Rundung des Standardfehlers vor der Multiplikation).
- **Zufallsgenerator:** die Normalzahlen kommen aus Reihen für Logarithmus und Kosinus nur mit Grundrechenarten (statt der exakten `decimal`-Fassung, 80 bis 140 µs je Zahl, und statt einer Näherung der Umkehrfunktion); sie stimmen mit der exakten Fassung auf 1e-13 überein (`tests/test_rng.py`) und sind schnell genug für 1000 Historien in 2 bis 12 Sekunden je Zeile.
- **Nicht umgesetzt:** die Wahl „ML oder REML“ als Regler in Experiment 4 (die App zeigt beide Anteile nebeneinander) und die Wahl „z oder t“ als Regler in Experiment 3 (die Tabelle zeigt beide Quantile).

## Modell

Zeilen: N Import-Container von m Kunden, im Mittel k̄ = N/m je Kunde (der Kunde jedes Containers wird gleichverteilt gezogen, die Kunden sind ungleich groß). Antwort: log-Verweilzeit,

y = ln 3,5 + 0,7·Zollhalt + 0,3·Wochenende + β_R·Rahmenvertrag + u_Kunde + 0,55·ε,   u_Kunde ~ N(0, σ_u²),  ε ~ N(0, 1),

mit Zollhalt 0,25 und Wochenende 2/7 je Container wie im Containergenerator der [`verweilzeit-demo`](https://github.com/sebastian-hanisch/verweilzeit-demo) (Import-Zweig dort neu geschrieben, nichts importiert). **Der Rahmenvertrag ist erfunden** (ein Kundenmerkmal, das der Generator nicht hat; Wirkung β_R = −0,30; Verteilung halb/halb, nur 2 Kunden oder stetig). Zollhalt und Wochenende wechseln innerhalb der Kunden. **Die Kundeneffekte werden in jeder Historie frisch gezogen** (Annahme: die Kunden sind eine Stichprobe aus einer Kundschaft); die zwölf festen Kundeneffekte des Generators sind eine einzige Ziehung (Experiment 5).

Zufallsachsenabschnitt-Modell mit V = σ²(I + λZZ'), λ = σ_u²/σ², Intraklassenkorrelation ρ = σ_u²/(σ_u² + σ²). Alle Formeln und Zahlen sind **eigene Nachrechnung in diesem Modell, bei diesen Parametern**; die Formeln stehen im 📐-Expander der App, dort auch die Konventionstabelle:

| Größe | Konvention hier | Andere Konvention |
|---|---|---|
| Teiler von σ̂² und σ̂_u² | REML: N − p; ausgewogen (MSB − MSW)/k | ML teilt durch N; Handbeispiel: σ̂_u² 20/3 (ML) gegen 11 (REML) |
| Parametrierung | λ = σ_u²/σ² | lme4 nutzt θ = σ_u/σ = √λ (nach dem Quellenbericht, nicht erneut geöffnet) |
| Intraklassenkorrelation | ρ = σ_u²/(σ_u² + σ²) | je nach Schätzer andere Zahl: Handbeispiel REML 11/15, ML 5/8 |
| Freiheitsgrade | naiv t(N − p), CR1 und REML t(m − 1), Kunden-Mittel-Methode t(m − 2) | für ein Kundenmerkmal im ausgewogenen Fall ist m − 2 exakt; Satterthwaite und Kenward-Roger nicht gerechnet |
| CR1 | Faktor m/(m − 1) · (N − 1)/(N − p) | CR0 ohne Faktor; CR2, CR3 nicht gerechnet; die Zählung CR0/CR1 ist hier über die Formel definiert, nicht aus einer Quelle belegt |
| „singulär“ | σ̂_u² = 0 (Randlösung λ = 0) | lme4-Sprachgebrauch: die Kovarianzmatrix der Zufallseffekte ist singulär; dasselbe, kein Absturz |
| Zähleinheit | Wiederholung = eine Historie (eine Tabelle aus m Kunden) | „Container“ nur, wo es dasteht |

## Methodik

- **Profilierte Rechnung selbst gerechnet** (`clg_mixed.py`, nur numpy und scipy): für festes λ ist die verallgemeinerte Rechnung eine p×p-Rechnung in Kundensummen (Woodbury); das profilierte ML- und REML-Kriterium hängt nur von λ ab (Bates u. a. 2015, Gleichungen 34 und 41 nach dem Quellenbericht, hier nachgerechnet). Die Suche über λ ≥ 0 löst die **analytische Ableitung** des Kriteriums (Nullstellensuche), der Rand λ = 0 wird über die Ableitung bei 0 festgestellt.
- **CR1**, **Kunden-Mittel-Methode** (Regression der Kundenmittel auf das Kundenmerkmal, t(m − 2)), **feste Kunden-Effekte** (Kunden-Dummies, nur für Merkmale innerhalb der Kunden; für ein Kundenmerkmal ist die Spalte eine Linearkombination der Dummies, der Koeffizient nicht schätzbar: die App zeigt den Rang).
- **Designeffekt:** exakte Varianz des OLS-Koeffizienten bei der wahren Kovarianz (Sandwich in Kundensummen, ohne N×N-Matrix) gegen die naive Formel; die wirksame Zahl unabhängiger Container ist N durch den Designeffekt.
- **Messung:** 1000 Historien je vorgerechnete Zeile (Datei `data/clg_frozen.json`, erzeugt von `tools/clg_make_frozen.py`, 20 Zeilen, kennzeichnet sich in der App als „vorgerechnet“) und eine **Live-Rechnung** mit wählbarer Wiederholungszahl (100 bis 1000, begrenzt auf etwa 2,5 Sekunden Rechenzeit). Der Zufall kommt aus dem ganzzahligen SplitMix64 (Zufalls-Seed); jede Historie hat einen eigenen aus dem Zufalls-Seed der Zeile abgeleiteten Startwert. Die vorgerechneten Zeilen sind deterministisch (nur Grundrechenarten); `python tools/clg_make_frozen.py --check` rechnet neu und vergleicht als Band.

## Experimente in der App

1. **Von Hand, drei Kunden:** das Handbeispiel (1, 3 | 5, 9 | 8, 10) und der Randfall (1, 3 | 2, 6 | 4, 4) in Brüchen: MSB, MSW, die Schätzer von σ_u² (REML und ML), das profilierte Kriterium über ρ, naiver und gemischter Standardfehler.
2. **Wie viele unabhängige Container?** Designeffekt exakt gegen die Formel 1 + (k̄ − 1)ρ, wirksame Zahl; für den Zollhalt das Verhältnis zur naiven Varianz.
3. **Welches Intervall deckt?** Überdeckung von sieben Verfahren (naiv, CR1 mit z und t(m−1), ML mit z, REML mit z und t(m−1), Kunden-Mittel-Methode), live und vorgerechnet; dazu der Zollhalt mit den festen Kunden-Effekten.
4. **Wenn auch das gemischte Modell am Rand liegt:** Anteil der Historien mit σ̂_u² = 0 (ML und REML) gegen die geschlossene Formel, Überdeckung bedingt auf den Rand, der Rang bei festen Kunden-Effekten.
5. **Die zwölf Kunden des Generators:** realisierte gegen nominale Streuung, Genauigkeit von σ̂_u² (Streuung über Historien gegen Kundenzahl und Container je Kunde), mit und ohne Rahmenvertrag im Modell.

## Befunde (gemessen, keine Behauptungen)

Alle Überdeckungen: 95-%-Intervall, Anteil der Historien, die den wahren Wert treffen; 1000 Historien je Zeile (vorgerechnet), Monte-Carlo-Fehler etwa 0,7 Prozentpunkte bei 95 % und 1,6 bei 50 %; **Unterschiede unter 1,4 Prozentpunkten werden nicht gedeutet** (Maßstab für den Vergleich zweier Zeilen dieser Reihe; Unterschiede zwischen zwei getrennten Messungen werden am Standardfehler der Differenz gemessen). Kundenmerkmal = Rahmenvertrag halb/halb, wenn nichts anderes steht; σ_u = 0,30.

| Einstellung | naiv | CR1, z | CR1, t(m−1) | ML, z | REML, z | REML, t(m−1) | Kunden-Mittel-Methode |
|---|---|---|---|---|---|---|---|
| m = 6, k̄ = 30 | 49,7 % | 82,9 % | 91,1 % | 80,4 % | 86,4 % | 93,5 % | 94,9 % |
| m = 12, k̄ = 30 | 50,8 % | 91,0 % | 94,5 % | 90,0 % | 93,6 % | 95,8 % | 95,7 % |
| m = 24, k̄ = 30 | 54,1 % | 92,8 % | 94,4 % | 92,4 % | 93,7 % | 94,7 % | 94,8 % |
| m = 48, k̄ = 30 | 51,8 % | 93,6 % | 94,5 % | 94,1 % | 94,5 % | 95,2 % | 95,4 % |
| m = 12, k̄ = 10 | 71,7 % | 89,3 % | 92,1 % | 88,8 % | 90,9 % | 94,2 % | 93,8 % |
| m = 12, k̄ = 120 | 26,9 % | 91,2 % | 93,9 % | 90,0 % | 92,2 % | 95,2 % | 95,4 % |
| m = 12, k̄ = 30, σ_u = 0 | 94,9 % | 90,9 % | 94,3 % | 95,2 % | 96,2 % | 97,8 % | 95,9 % |
| m = 12, k̄ = 30, σ_u = 0,10 | 85,0 % | 92,0 % | 94,2 % | 92,3 % | 93,7 % | 95,8 % | 94,7 % |
| m = 12, k̄ = 30, σ_u = 0,20 | 60,4 % | 87,9 % | 91,7 % | 87,3 % | 90,1 % | 93,6 % | 94,6 % |
| m = 12, k̄ = 30, σ_u = 0,45 | 40,1 % | 91,0 % | 94,2 % | 90,4 % | 92,9 % | 96,9 % | 96,2 % |
| m = 12, k̄ = 30, Merkmal bei nur 2 Kunden | 49,4 % | 75,8 % | 79,8 % | 88,5 % | 91,7 % | 95,4 % | 95,0 % |
| m = 12, k̄ = 30, Merkmal stetig | 52,7 % | 86,8 % | 89,1 % | 89,6 % | 92,0 % | 94,5 % | 94,2 % |
| m = 6, k̄ = 30, Merkmal stetig | 51,5 % | 74,6 % | 82,7 % | 81,8 % | 87,6 % | 94,0 % | 94,8 % |

(Tabelle: Rahmenvertrag; jede Zeile steht in `data/clg_frozen.json`, die Zählungen prüft `tests/test_claims.py`.)

| Frage | Befund | Voraussetzung | Beleg |
|---|---|---|---|
| Designeffekt im Vehikel | exakt 7,82 im Median (5 bis 95 % der Zuordnungen: 7,69 bis 7,99); die Formel 1 + (k̄ − 1)ρ mit mittlerer Kundengröße gibt 7,65; wirksam etwa 46 von 360 Containern | 12 Kunden, 30 Container je Kunde im Mittel, Rahmenvertrag halb/halb, ρ = 0,229, Median über 2000 Zuordnungen | `test_claims.py::test_designeffekt_im_vehikel` |
| Designeffekt für den Zollhalt (Merkmal wechselt zufällig im Kunden) | wahre Varianz gegen erwartete naive Varianz im Median zwischen 0,95 und 1,00 über 36 Einstellungen (m 6, 12, 48; k̄ 10, 30, 120; σ_u 0, 0,10, 0,30, 0,45) | Median über je 1000 Zuordnungen | `test_claims.py::test_designeffekt_zollhalt_ueber_36_einstellungen` |
| Naives Intervall und Zollhalt | 94,4 bis 96,4 % in allen 20 vorgerechneten Zeilen; REML t(m−1) 95,5 bis 99,2 %, feste Kunden-Effekte 94,1 bis 96,6 % | die 20 Zeilen der Messung | `test_claims.py::test_zollhalt_naiv_nicht_zu_schmal_in_allen_zeilen` |
| Breite beim Zollhalt, m = 12, k̄ = 30 | REML mit z 0,90-fach der naiven Breite, feste Kunden-Effekte 0,91-fach, REML mit t(m−1) 1,01-fach, bei σ_u = 0,45 REML z 0,81-fach | 1000 Historien, Mittel der Breiten | `test_claims.py::test_breiten_zollhalt` |
| Anteil am Rand | σ_u = 0: REML 54,6 %, ML 72,2 %; σ_u = 0,10: 11,1 % und 19,4 %; σ_u = 0,20: 0,7 % (7 Historien, zu wenige zum Deuten) und 1,4 %; m = 6, k̄ = 30, σ_u = 0,30: 2,2 % (22 Historien) | 1000 Historien, m = 12, wenn nichts anderes steht | `test_claims.py::test_anteil_am_rand` |
| Überdeckung am Rand, σ_u = 0,10 | REML t(m−1) 88,3 % (111 Historien am Rand) gegen 96,7 % (889 Historien sonst); naiv am Rand 85,6 %, CR1 t(m−1) 82,9 % | 1000 Historien | `test_claims.py::test_ueberdeckung_bedingt_auf_den_rand` |
| Rand im ausgewogenen Fall | Formel P(F ≤ 1/(1 + kλ)): 27,1 % (m = 10, k = 10, ρ = 0,05, Kundenmerkmal), 55,8 % (ρ = 0), 9,0 % (m = 30); Simulation stimmt innerhalb von vier Standardfehlern | je 1500 Wiederholungen, ausgewogen | `test_oracle_cluster_gemischt.py::test_anteil_am_rand_gleicht_der_f_formel`, `test_claims.py::test_formelwerte_fuer_den_rand` |
| Streuung von σ̂_u² | m = 12: 0,045 (30 Container je Kunde), 0,054 (10), 0,041 (300); m = 48: 0,022 (30); m = 6: 0,073 (30); die Grenze σ_u²·√(2/(m−1)) für sehr viele Container ist 0,038 bei m = 12 und 0,019 bei m = 48 (so gilt sie ohne Kundenmerkmal im Modell; die Messreihe hat den Rahmenvertrag im Modell, dort ist sie σ_u²·√(2/(m−2)) = 0,040 bei m = 12, passend zu 0,041 bei 300 Containern je Kunde): mehr Kunden helfen, mehr Container kaum | σ_u² = 0,09, 1000 Historien, REML | `test_claims.py::test_streuung_der_varianzkomponente` |
| Rahmenvertrag fehlt im Modell | σ̂_u² im Mittel 0,117 statt 0,091 (mit dem Merkmal); erwartet etwa 0,1125 (0,09 + 0,30²·0,25, für gleich große Kunden) | m = 12, k̄ = 30, 1000 Historien | `test_claims.py::test_varianzkomponente_ohne_kundenmerkmal` |
| Die zwölf Kunden des Generators | Streuung 0,379 statt nominal 0,30 (Intraklassenkorrelation 0,322 statt 0,229); in 9,3 % von 20000 Ziehungen mit nominal 0,30 ist die Stichproben-Streuung von zwölf Kunden mindestens so groß | feste Zahlen des Generators, Zufalls-Seed 5000 | `test_claims.py::test_zwoelf_kunden_des_generators` |
| Handbeispiel (Brüche) | MSB 26, MSW 4; REML σ̂_u² = 11, ML 20/3; ρ̂ 11/15 und 5/8; gemischte Varianz des Mittels 13/3 gegen naiv 32/15; Halbbreiten t(2): 8,96, naiv t(5): 3,75 | drei Kunden mit je zwei Containern | `test_mixed.py::test_hand1_*` |
| Randfall (Brüche) | MSB 8/3, MSW 10/3; Schätzer REML −1/3, ML −7/9: beide negativ, Rand; σ̂² = 46/15 (REML) bzw. 23/9 (ML) | y = (1, 3 \| 2, 6 \| 4, 4) | `test_mixed.py::test_hand2_*` |
| Designeffekt von Hand | vier Kunden à drei Container, ρ = 1/4: wahre Varianz 2/3, Designeffekt 3/2 = 1 + (k − 1)ρ, wirksam 8; Merkmal mittelwertfrei im Kunden: 1/8, Faktor 3/4 | σ² = 1, σ_u² = 1/3 | `test_mixed.py::test_hand3_*` |
| CR1 gegen Kunden-Mittel-Methode von Hand | CR1-Varianz 11/30, Welch-Varianz der Kundenmittel 1/2; Halbbreiten 1,95 (naiv), 1,93 (CR1, t(3)), 3,04 (Kunden-Mittel-Methode, t(2)); beide Standardverfahren sind bei vier Kunden um etwa 36 % schmaler als die Kunden-Mittel-Methode (der Bezugswert hier, im ausgewogenen Fall mit t(m−2) richtig) | dieselben vier Kunden, Steigung 3 | `test_mixed.py::test_hand4_*` |
| Kern gegen geschlossene Form | REML und ML stimmen in 300 ausgewogenen Datensätzen (158 mit negativem Schätzer, also am Rand) auf 1e-13 mit der Bruchrechnung überein (Messung dieser README, nicht im Test; der Test prüft 120 Datensätze auf 1e-7) | m 3 bis 8, k 2 bis 5, ganze Zahlen | `test_oracle_cluster_gemischt.py::test_reml_und_ml_gleichen_der_geschlossenen_form_auch_am_rand` |
| Kern gegen dichte Matrix | das profilierte Kriterium unterscheidet sich von der N×N-Fassung nur um eine Konstante (1e-9, ML und REML), GLS und Q bei festem λ auf 1e-10 | 6 Kunden, 27 Container | `test_oracle_cluster_gemischt.py::test_profiliertes_kriterium_*` |

## Zahl → Test (Band oder exakt)

Nicht jede Zahl der README steht in einem Test; diese Tabelle sagt, wo sie geprüft wird.

| Zahl in der README | Geprüft in | Art |
|---|---|---|
| Überdeckungen der Tabelle oben, Rand, Breiten, σ̂_u²-Streuungen | `tests/test_claims.py` liest `data/clg_frozen.json` und sucht jede Zahl im Text; `tools/clg_make_frozen.py --check` (Test `test_vorgerechnete_zeilen_stimmen_mit_neuer_rechnung`, drei Zeilen) rechnet neu | exakt gegen die Datei, Neu-Rechnen als Band (Zählungen ± 2) |
| Designeffekt 7,82, Formel 7,65, wirksam 46, bis zu 7 %, Zollhalt 0,95 bis 1,00 und 0,99 | `test_claims.py` (Neu-Rechnen mit festem Zufalls-Seed) | Band (± 0,03, wirksam ± 1) |
| 9,3 %, 0,379, 0,322, 0,229 | `test_claims.py` | Band |
| Bruchwerte der Handbeispiele | `tests/test_mixed.py` | exakt in Brüchen |
| 300 Datensätze, 1e-13 | Messung dieser README (nicht im Test; der Test prüft 120 auf 1e-7) | |
| Zahlen der früheren Messung (93,0 %, 52,2 %, 12,5 %, 29,3 %, 3,76) | nur zitiert, nicht im Test | |
| Laufzeit „2 bis 12 Sekunden je Zeile“, „etwa 2,5 Sekunden“ | Messung dieser README auf einem Windows-Rechner unter wechselnder Last, nicht im Test (der Test `test_teuerster_fall_am_regler_maximum_mit_zeitgrenze` hat eine Zeitgrenze, keine Zeitzahl) | |

## Ehrliche Grenzen

- **Nur ein Zufallsachsenabschnitt je Kunde.** Zufallssteigungen, mehrere Gruppierungsfaktoren (Kunde und Schiff) und geschachtelte Cluster sind nicht gerechnet; ebenso verallgemeinerte gemischte Modelle (Standgeld, Standtage je Kunde), Satterthwaite- und Kenward-Roger-Freiheitsgrade, CR2 und der Wild-Cluster-Bootstrap (Cameron/Miller nennen ihn als Abhilfe bei wenigen Clustern; hier nicht gemessen).
- **Nur Import-Container, nur log-Skala.** Bei Export, Umschlag und Leer ist der Kundenanteil kleiner oder die Streuung anders; nicht gemessen.
- **Kunden frisch gezogen, nicht fest:** für einen festen, endlichen Kundenstamm (die zwölf Kunden des Generators als Grundgesamtheit) ändert sich die Fragestellung; nicht gemessen.
- **Nur 6 bis 48 Kunden.** Unter 6 Kunden (Cameron/Miller nennen 5 bis 6 Cluster als am schwierigsten) ist nicht gemessen; extreme Größenunterschiede (ein Kunde mit der Hälfte der Container) ebenso; die Kundengrößen sind nur die zufälligen der gleichverteilten Zuordnung.
- **Der Rahmenvertrag ist erfunden.** Wirkung −0,30 und die drei Verteilungen sind Annahmen; „CR1 reicht nicht“ gilt mit der Verteilung des Merkmals (halb/halb 94,5 %, 2 von 12 Kunden 79,8 %, stetig 89,1 %), nicht allgemein.
- **Monte-Carlo-Fehler:** 1000 Historien geben bei 95 % etwa 0,7 Prozentpunkte; Anteile unter 1 % (z. B. 0,7 % am Rand bei σ_u = 0,20) beruhen auf wenigen Historien und werden nicht gedeutet. Die Live-Rechnung (Voreinstellung 200 Historien) hat einen Fehler von etwa 1,5 Prozentpunkten.
- **Die Formel 1 + (k̄ − 1)ρ mit mittlerer Kundengröße** weicht im Vehikel um bis zu 7 % vom exakten Designeffekt ab (2000 Zuordnungen, Median der Abweichung unter 3 %; `test_claims.py`); die App zeigt beide Werte.
- **Der Wechsel zu festen Kunden-Effekten** ist für ein Merkmal innerhalb der Kunden gemessen (Überdeckung 94,1 bis 96,6 %); für den Rahmenvertrag nicht schätzbar. „Gemischt beginnen, bei Singularität fest“ ist hier also nur für Merkmale innerhalb der Kunden eine mögliche Regel; ihre Überdeckung bedingt auf den Rand wurde nicht gemessen.
- **Quellen:** Cameron/Miller, Bates u. a. und Oberpriller u. a. sind **nach dem Quellenbericht** zu dieser Demo zitiert (Wortlaut gelesen, hier nicht erneut geöffnet); die Zahlen aus Cameron/Miller stammen aus einem anderen Erzeugermodell und sind hier **nicht nachgerechnet** (nur Richtung und Größenordnung werden verglichen: CR mit Normalquantil überlehnt bei wenigen Clustern; hier deckt CR1 mit z bei 6 Kunden 82,9 % und bei 12 Kunden 91,0 %). Der Designeffekt 1 + (k − 1)ρ ist eigene Rechnung (Quelle nicht gelesen).

## Quellen (Grad der Einsicht)

| Quelle | Grad | Genutzt für |
|---|---|---|
| Bates, Mächler, Bolker, Walker (2015): Fitting Linear Mixed-Effects Models Using lme4, Journal of Statistical Software 67(1) | Wortlaut gelesen (nach Quellenbericht) | profiliertes Kriterium (Gleichungen 34 und 41), Aussage, dass singuläre Anpassungen in der Praxis häufig sind, und dass lme4 bewusst keine p-Werte ausgibt (Freiheitsgrade-Näherungen seien ad hoc); in eigenen Worten |
| Cameron, Miller (2015): A Practitioner's Guide to Cluster-Robust Inference, Journal of Human Resources 50(2) | Wortlaut gelesen (nach Quellenbericht) | bei wenigen Clustern sind cluster-robuste Varianzen zu klein und Normalquantile lehnen zu oft ab; Abhilfen t(G − 1), Kleinstichprobenkorrektur, Wild-Cluster-Bootstrap; Zahlen nicht übernommen |
| Oberpriller, de Souza Leite, Pichler (2022): Fixed or random? …, Ecology and Evolution 12 | Wortlaut gelesen (nach Quellenbericht) | bei wenigen Gruppenstufen kann ein festes Modell das gemischte ersetzen; Empfehlung, gemischt zu beginnen |
| Bates, Kliegl, Vasishth, Baayen (Parsimonious Mixed Models) und Matuschek u. a. (Balancing Type I Error and Power) | Abstract | maximale Zufallsstruktur kostet Teststärke und konvergiert oft nicht (nur als Hinweis; die Zufallsstruktur ist hier nicht Thema) |
| Barr u. a. (2013) | nur über Matuschek u. a. (Suchtreffer) | nicht genutzt |

## Tests

Die Suite hat 165 Tests (Laufzeit lokal rund 75 Sekunden bei geringer Last auf einem Windows-Rechner, nicht im Test festgeschrieben): Handbeispiele in Brüchen (`test_mixed.py`), Orakel (`test_oracle_cluster_gemischt.py`: geschlossene Form in Brüchen auch am Rand, dichte Matrix ohne Optimierer, CR1 mit Schleife, Designeffekt gegen Simulation, Rand gegen die F-Verteilung, Streuung des ANOVA-Schätzers, Kunden-Mittel-Methode exakt), Eigenschaften (`test_eigenschaften_cluster_gemischt.py`: Skalierung, Verschiebung, Umbenennen und Umsortieren, Gleichheit von GLS und OLS im ausgewogenen Fall, CR1 gleich HC1 bei Clustern der Größe 1, Designeffekt, Rang), Randfälle (ein Kunde, jeder Kunde mit einem Container, alle Werte gleich, leere Kunden, Rangmangel, das Optimum zwischen 0 und dem ersten Gitterpunkt), Zufallsgenerator (`test_rng.py`), Vehikel, Verdikte (je Preset die erwartete Zeile), Claims, Presets, Layout (jede Änderung eines Reglers hebt die Markierung auf), AppTest-Rauchtests (jeder Knopf, jeder Regler an den Grenzen, jedes Preset, Permalink, teuerster Fall mit Zeitgrenze), Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Ergebnis, Kernfragen, fünf Experimente, Grenzen, 📐-Expander) |
| `clg_mixed.py` | Kern: profiliertes ML und REML mit Ableitung, CR1, Kunden-Mittel-Methode, feste Effekte, Designeffekt, geschlossene Formeln in Brüchen |
| `clg_vehicle.py` | Vehikel: Import-Container mit Kunden, Historien, Simulation, Designs |
| `clg_rng.py` | ganzzahliger Zufallsgenerator SplitMix64 mit Normalzahlen ohne C-Bibliothek |
| `clg_evaluation.py` | Eingabe und Auswertung des Handbeispiels, Verdikte, vorgerechnete Zeilen |
| `clg_constants.py` | Regler-Grenzen, Presets, die Zeilen der vorgerechneten Messung |
| `clg_presets.py` | Permalink (Begrenzen, Einrasten), Presets, Zufalls-Seed-Knopf |
| `clg_layout.py` | fester Kopf (Stylesheet) |
| `clg_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `data/clg_frozen.json` | vorgerechnete Messung (20 Zeilen mit je 1000 Historien, dazu σ̂_u² mit und ohne Rahmenvertrag) |
| `tools/clg_make_frozen.py` | erzeugt die vorgerechnete Messung; `--check` vergleicht als Band |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit-Bibliotheken (ungepinnt), dazu pytest |
| `tests/` | siehe Abschnitt Tests |

## Bewusst nicht umgesetzt

- Zufallssteigungen, mehrere Gruppierungsfaktoren, geschachtelte Cluster, verallgemeinerte gemischte Modelle (Standgeld, Standtage), Satterthwaite und Kenward-Roger, CR2 und CR3, Wild-Cluster-Bootstrap, gruppierte Kreuzvalidierung (Kunden zusammenlassen): nicht gerechnet, jeweils offene Fragen.
- Eine Wahl der Zufallsstruktur nach Kriterium (AIC, BIC): gehört zur Modellwahl ([`modellwahl-demo`](https://github.com/sebastian-hanisch/modellwahl-demo)).
- PDF-Export der Ergebnisse.
- Echtdaten: alle Kunden und Container sind erfunden.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Regression und GLM: wann die Gleichung trügt](https://sebastianhanisch.net/konzepte-regression.html).

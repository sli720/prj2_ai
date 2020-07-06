Sämtliche Änderungen am Sourcecode im SS20 wurden mit "#UPDATE_SS20: <text>" kommentiert und erklärt.

Die Files "FilterDatasets.py" und "CorrelationHeatmap.py" wurden neu hinzugefügt. Ersteres
filtert ein Dataset und lässt nur Datenpunkte vorgegebener Labels übrig. Zweiteres dient
zum Berechnen von Korrelationen und darstellen dieser als Heatmaps.

Soll die gewählte Struktur beibehalten werden, so muss im Ordner "Daten/" das Dataset
eingefügt werden. Innerhalb dieses Ordners befinden sich die weiteren Verzeichnisse
"NewProject" und "OldProject". Hier werden train- und test sets abgespeichert, sowie
die fertigen Modelle.

Alle anderen Informationen zur Bedienung des Sourcecodes sind im File "ReadMe.txt" ersichtlich.

# Small helper utility to show the first 10 rows of a file
# Our input files are so big that notepad / notepad++ can't open the file
# The script can therefore be used to view these files (e.g.: on Windows)


file_path = "M:\\output\\merged.csv"

index = 0
with open(file_path, "r") as fobj:
        while True:
            line = fobj.readline().strip()
			
            if line: 
                print(line)
            else:
                break
            index += 1
            if index == 30:
                break
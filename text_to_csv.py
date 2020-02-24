import csv
import os

with open('C:\\train\\txt') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('train.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax'))
        writer.writerows(lines)

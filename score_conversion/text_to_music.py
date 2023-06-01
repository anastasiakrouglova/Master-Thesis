# run "python score_conversion/text_to_music.py > score_conversion/piano_score.ly" in terminal
# to create .ly file
# run "export PATH=/Applications/lilypond-2.24.1/bin/:$PATH" if lilypond not found in terminal after installation
# then, run execute.py


# documentation: https://lilypond.org/doc/v2.22/Documentation/notation/writing-pitches
    

#open text file in read mode
text_file = open("score_conversion/notes.txt", "r")
#read whole file to a string
data = text_file.read()

#close files
text_file.close()

    
staff = "{\n\\new PianoStaff << \n"
staff += "  \\new Staff {" + data  + "}\n"  # upper_staff "c'' b' ais' a' gis' g' fis' f'"
# staff += "  \\new Staff { \clef bass " + lower_staff + "}\n"  
staff += ">>\n}\n"

title = """\header {
  title = "Music extraction"
  composer = "Anastasia Krouglova using Python"
  tagline = "Copyright: Anastasia Krouglova"
}"""

print(title + staff)

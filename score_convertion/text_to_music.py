# run 'python text_to_music.py > piano_score.ly' in terminal
# to create .ly file
# run export PATH=/Applications/lilypond-2.24.1/bin/:$PATH if lilypond not found in terminal after installation
# then, lilypond piano_score.ly to create pdf

# documentation: https://lilypond.org/doc/v2.22/Documentation/notation/writing-pitches

# char2notes = { 
#   ' ':("a4 a4 ", "r2 "),
#   'a':("<c a>2 ", "<e' a'>2 "),
#   'b':("e2 ", "e'4 <e' g'> "),
#   'c':("g2 ", "d'4 e' "),
#   'd':("e2 ", "e'4 a' "),
#   'e':("<c g>2 ", "a'4 <a' c'> "),
#   'f':("a2 ", "<g' a'>4 c'' "),
#   'g':("a2 ", "<g' a'>4 a' "),
#   'h':("r4 g ", " r4 g' "),
#   'i':("<c e>2 ", "d'4 g' "),
#   'j':("a4 a ", "g'4 g' "),
#   'k':("a2 ", "<g' a'>4 g' "),
#   'l':("e4 g ", "a'4 a' "),
#   'm':("c4 e ", "a'4 g' "),
#   'n':("e4 c ", "a'4 g' "),
#   'o':("<c a g>2  ", "a'2 "),
#   'p':("a2 ", "e'4 <e' g'> "),
#   'q':("a2 ", "a'4 a' "),
#   'r':("g4 e ", "a'4 a' "),
#   's':("a2 ", "g'4 a' "),
#   't':("g2 ", "e'4 c' "),
#   'u':("<c e g>2  ", "<a' g'>2"),
#   'v':("e4 e ", "a'4 c' "),
#   'w':("e4 a ", "a'4 c' "),
#   'x':("r4 <c d> ", "g' a' "),
#   'y':("<c g>2  ", "<a' g'>2"),
#   'z':("<e a>2 ", "g'4 a' "),
#   '\n':("r1 r1 ", "r1 r1 "),
#   ',':("r2 ", "r2"),
#   '.':("<c e a>2 ", "<a c' e'>2")
# }


# txt = "Love one another and you will be happy. It is as simple as that."

# upper_staff = ""
# lower_staff = "" 
# for i in txt.lower():
#     (l,u) = char2notes[i]
#     upper_staff += u
#     lower_staff += l
    

#open text file in read mode
text_file = open("notes.txt", "r")
#read whole file to a string
data = text_file.read()
#close file
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

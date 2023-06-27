import os

# Or do it manually: see text_to_music.py comments

# go to the right folder
exit_status = os.system("cd /Users/nastysushi/Mirror/_MULTIMEDIA/THESIS/thesis/github/code/score_conversion")
# to create .ly file
exit_status = os.system("python ./code/score_conversion/text_to_music.py > ./code/score_conversion/piano_score.ly")
# run this command in case lilypond is not found in terminal after installation
exit_status = os.system("export PATH=/Applications/lilypond-2.24.1/bin/:$PATH")
# to create .pdf file
exit_status = os.system("lilypond ./code/score_conversion/piano_score.ly")


# Check the return value
if exit_status == 0:
    print("Command succeeded")
else:
    print("Command failed")
import os

# go to the right folder
exit_status = os.system("cd /Users/nastysushi/Mirror/_MULTIMEDIA/THESIS/thesis/github")
# to create .ly file
exit_status = os.system("python text_to_music.py > piano_score.ly")
# run this command in case lilypond is not found in terminal after installation
exit_status = os.system("export PATH=/Applications/lilypond-2.24.1/bin/:$PATH")
# to create .pdf file
exit_status = os.system("lilypond piano_score.ly")


# Check the return value
if exit_status == 0:
    print("Command succeeded")
else:
    print("Command failed")
1.  The data floder includes input and output,      
you can download it from here:https://drive.google.com/drive/folders/18xgkKy4pcxddSPDr99GNYswu4CZa_nwD?usp=sharing  
the input is the raw database, calls 1410-15.csv
the output includes 2 tables:    
1.the games table:   1410-15result.csv    (the results of 'games_flow2.ipynb' in folder 'notebooks'; And if you run 'player_flow.ipynb' in folder'notebooks',it also can be the input file,just change the filepath)
2.the players table:   df_final.csv    (the results of 'player_flow.ipynb')

2.  the "models" folder includes the original ipynb files which doesn't warpped with Prefect

3.  the "notebooks" folder includes 2 ipynb files wrapped with prefect, I recommand to run this but not the main.py cause this is more clear
This is what should do before run the file in "notebooks",basically change the filepaths:
1.open "games_flow2.ipynb", and go to " deploy_analysis" function and change the path , then run it ,you will get a table contains all the games' details
2.open "player_flow.ipynb",go to the last cell and change the csv path
(the input should be the results of "games_flow2.ipynb",but you can also directly choose the "1410-15result.csv"file you downloaded from the link at the beginning so that you dont have to wait a long time.)



4.  The main.py script contains the Prefect pipeline that replicates the functionality of the two Jupyter notebooks in the notebooks folder. It is essentially a consolidated version of those notebooks, rewritten as a standalone Python script.But i almost just copy from my "notebooks" folder so im not sure this can run the right answer

But if still want to execute the main.py, you have to change the file path before:
Scroll down to the bottom of "main.py" and locate the "deploy_analysis" function. Replace the engine_path with the path to your local Stockfish executable, and change the "csv_file_path" to the path of '1410-15.csv'.
At the same time, update the path in the "read_csv" function inside "player_analysis" function to the absolute path of '1410-15result.csv'.



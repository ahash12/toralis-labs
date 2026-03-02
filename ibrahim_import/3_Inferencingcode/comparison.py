##Inference only (new geometry → WSS prediction)
"""
python inference.py ^
  --ckpt "C:\\Users\\radie\\Desktop\\trainingprocess\\TRAIN_OUT\\best_model.pt" ^
  --msh  "C:\\path\to\new_case.msh" ^
  --out_dir "C:\path\to\OUT" ^
  --re 400 ^
  --export_vtp --export_csv --save_pred_pt

  """


##Validation mode (your chosen #2): new geometry + Fluent ground truth 
# Run Fluent on the new geometry, export wall_data_Re400.csv, then:

"""
python inference.py ^
  --ckpt "...\best_model.pt" ^
  --msh  "...\new_case.msh" ^
  --out_dir "...\OUT" ^
  --re 400 ^
  --cfd_wall_csv "...\wall_data_Re400.csv" ^
  --export_vtp --export_csv

  """
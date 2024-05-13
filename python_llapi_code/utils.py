def save_dataset(path):
  with open(path, "wb") as f:
    pickle.dump(data, f)
  
  
def load_dataset(path):
  with open(path, "rb") as f:
    data = pickle.load(f)
  return data
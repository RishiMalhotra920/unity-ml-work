

class RolloutBuffer:
  def __init__(self):
    self.rollout = []
    self.end = False
  
  def add(self, state, action, reward, state_prime):
    self.rollout.append((state, action, reward, state_prime))
    
  def set_end(self, end):
    self.end = end
  
  def get_end(self):
    if self.end == None:
      raise Exception("end not set")
    
    return self.end
  
  def get_last_state(self):
    return self.rollout[-1][3]
  
  def reset(self):
    self.rollout = []
    self.end = None
  
  def get_rollout(self):
    return self.rollout
  
  def __len__(self):
    return len(self.rollout)
  
  def __getitem__(self, key):
    return self.rollout[key]
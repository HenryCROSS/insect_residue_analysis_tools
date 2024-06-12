class Test:
  def __init__(self):
    self.a = 1

t = Test()

def change(tt):
  tt.a = 2

print(t.a)

change(t)

print(t.a)
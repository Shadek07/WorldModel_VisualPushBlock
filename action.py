
action_indexes = [3, 6, 1, 2, 5, 4] #(0.33 to 0.5), (-0.163, 0.33) so on
action = -0.47
x = 0.5
action_index = 4 #for value range [-1,0,65]
for i in range(6):
    if x-0.167 < action:
      action_index = action_indexes[i]
      break
    x = x-0.167 #1/6 = 0.167, action value range = (1-(-1)) = 2
print(action_index)
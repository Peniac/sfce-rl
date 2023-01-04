import pickle


with open('../results/mean_rewards_50', 'rb') as obj:
    data = pickle.load(obj)

data.plot()

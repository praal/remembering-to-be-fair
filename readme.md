# Remembering to Be Fair: Non-Markovian Fairness in Sequenctial Decision Making
Code for the submission.

### Instructions to run DQN approaches for resource allocation env:
#### Run DQN with full representation: 
python dqn-donut.py -ep 1000 -nexp 10 -sm binary

#### Run DQN with FairQCM:
python dqn-donut.py -ep 1000 -nexp 10 -sm binary -cf True

#### Run DQN with Min representation: 
python dqn-donut.py -ep 1000 -nexp 10 -sm binary-reset

#### Run DQN with reset representation: 
python dqn-donut.py -ep 1000 -nexp 10 -sm binary-equal

#### Run DQN with RNN: 
python dqn-recurrent.py -ep 1000 -nexp 10 -sm rnn

### Instructions to run DQN approaches for lending env:
#### Run DQN with full representation: 
python dqn-lending.py -ep 1000 -nexp 10 

#### Run DQN with FairQCM:
python dqn-lending.py -ep 1000 -nexp 10 -cf True

#### Run DQN with Min representation: 
python dqn-lending.py -ep 1000 -nexp 10 -sm reset

#### Run DQN with RNN: 
python dqn-recurrent-lending.py -ep 1000 -nexp 10 -sm rnn

### Run QLearning:
python ql.py


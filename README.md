# Artificial Intelligence 2 (AI2) Project

## The Ludo Game

### Game rules

- Always four players.
- A player must roll a 6 to enter the board.
- Rolling a 6 does not grant a new dice roll.
- Globe positions are safe positions.
- The start position outside each home is considered a globe position
- A player token landing on a single opponent token sends the opponent token home if it is not on a globe position. If the opponent token is on a globe position the player token itself is sent home.
- A player token landing on two or more opponent tokens sends the player token itself home.
- If a player token lands on one or more opponent tokens when entering the board, all opponent tokens are sent home.
- A player landing on a star is moved to the next star or directly to goal if landing on the last star.
- A player in goal cannot be moved.

## Report

The AI2 course is evaluated by your handing in a "conference paper" describing your work. You must use the format kits provided on BlackBoard, and your paper must be no more than 11 pages long including figures.

Your paper will have en following sections. The % indicates roughly how much of your paper allowance should be used for each main section (they are guidelines, not rules.).

If you do several different experiments, you may end up repeating the “Methods- Results-Discussion” pattern for each experiment instead of doing all the method description first, then all the results, etc.: that’s OK.

### Introduction 5 %

Here you motivate the problem and describe what kind of methods you want to use and why!. In a real conference paper, this would be longer because the authors have to explain to other people why the problem is interesting. Here you are working with Ludo because that is the assigned task. You should introduce the content of your paper, but you needn’t explain why Ludo is an interesting problem.

### Methods 40 %

The goal for this section is the describe what you have done in enough detail that an intelligent and informed reader could repeat it and be sure they had done the same as you did. You don’t need to describe or explain standard techniques, such as Backpropagation or GA operation: give a citation to a textbook or reference work instead. For example, “I used a standard single population GA [cite textbook ] with rank-based selection, one-point crossover, standard mutation and generational replacement with 10% elitism.” would suffice to describe your basic GA method; of course you then have to give full details of the parameter choices (population size, probabilities, ...) and of the representation your GA was using.
You do need to describe your representation of the problem and how you encoded it (and why you chose that representation); you should describe any non-standard choices for operators or procedures; you should state what choices you made for the parameters (e.g. learning rate) and how you chose them (e.g. preliminary test, found in a paper (cited), guessed, etc.); you should describe the tests you conducted to ensure your code works as you claim. Describe what experiments you did.
You are expected to describe one AI method quite completely (your “own” one) and a second one much more briefly. The second one is the one you will be comparing against and will have been implemented, and described in detail, by someone else (e.g. a partner in your group). Remember to cite and/or acknowledge the source for this second method.
The two methods can be different algorithms, e.g. GA vs. Q-learning, or they can be two instances of the same algorithm using different game representations. You are free to compare more methods or to try to isolate important factors if you wish, but do not exceed the page limit of the paper format.

### Results 20 %

You describe what your measurements were, and what data you collected, in as neutral a manner as possible. You may wish to measure performance in Ludo: describe how you choose to do that, then present the performance data from your own AI method and from the comparison (second) one. Use comparisons with the Random and SemiSmart players as well, where appropriate.

### Analysis and Discussion 25 %

Here you give interpretation and analysis of the results in the previous section. Use statistics as needed. This is where you can conclude whether your method works better than the others, for example (if the data you collected shows this). You can also discuss the reasons for the results you obtained here.

### Conclusion 5 %

Report the main conclusions of your study. If you wish, discuss what you might do
differently or what is the next major step you would take in the investigation.

### Acknowledgements 5 % (inc. bibliography)

Thank the people who have helped you. Specifically, the person(s) from whom you obtained the comparison AI method (the “second method”) above. If you asked people for comment on your draft paper – which is a good idea – then acknowledge them here too.

**Bibliography**

The list of material you cited, with sufficient detail that a reader can find it for themselves. This list need not be long, for this paper.

### Major mistakes people make

- Methods not complete – there are parameters/procedures/choices not described.
- Conclusions are too enthusiastic and not supported by the data.
- Wasting space reciting textbook material.

### Submission

- Submit 1 pdf file by 23:59 on 25.05.2020.
- Please name your pdf file as "firstname_lastname_report_AI2".
- Submit via Blackboard (SDU Assignment).
- Check submissions for plagiarism using SafeAssign.

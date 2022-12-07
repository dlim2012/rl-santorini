

from board_games.Santorini.agents import RandomAgent, MiniMaxAgent
from board_games.Santorini.board import GameBoard

def main():
    print('main')
    agents = [MiniMaxAgent(maxDepth=2), MiniMaxAgent(maxDepth=1)]

    counts = [0] * len(agents)
    for i in range(1000):
        game_board = GameBoard(
            agents=agents,
            #print_simulation=True
        )

        game_board.reset()
        play = game_board.play()
        result = next(play)
        print(result, end=' ')
        counts[result] += 1
        print(counts[0] / sum(counts))
        #if result == 1:
        #    break
    print()
    print(counts)

if __name__ == '__main__':
    main()
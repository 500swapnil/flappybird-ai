from itertools import cycle
import random
import sys
import math
import numpy as np
from collections import namedtuple
from itertools import count
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import pygame
from pygame.locals import *
from tensorflow.keras.models import load_model
import os, pickle
import copy

FPS = 1000
SCREENWIDTH  = 288
SCREENHEIGHT = 512
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
IM_WIDTH = 0
IM_HEIGHT = 1
# image, Width, Height
PIPE = [52, 320]
PLAYER = [34, 24]
BASE = [336, 112]
BACKGROUND = [288, 512]
folder = './'
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

total_models = 40
current_pool = []
fitness_pool = np.zeros(total_models)
input_dim = 5
best_model_weights = []
best_fitness = -10
gen = 1
mutateProb = 0.15
total_gen = 40
load_saved_pool = False
save_current_pool = True
take_best_two = [] # [load_model("best_model_705_2.h5"), load_model("best_model_654_2.h5")]
best_model = load_model("best_model.h5")

def save_pool():
    for i in range(total_models):
        current_pool[i].save_weights("SavedModels/model_{0}.h5".format(i+1))

def create_model(input_dim):
    model = Sequential()
    model.add(Dense(12,  input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(24))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    model.compile(loss='mse', optimizer="adam")
    return model

def initialize_models():
    global current_pool, total_models
    if best_model is not None:
        print("Testing Existing Model")
        total_models = 1
        current_pool.append(best_model)
        return

    for i in range(total_models):
        current_pool.append(create_model(input_dim))

    if load_saved_pool:
        print("Loading Saved pool")
        for i in range(total_models):
            current_pool[i].load_weights("SavedModels/model_{0}.h5".format(i+1))
    elif len(take_best_two) == 2:
        print("Using best 2 models")
        next_gen_weights = []
        current_pool[0] = take_best_two[0]
        current_pool[1] = take_best_two[1]
        for i in range(total_models // 2):
            child1, child2 = crossover(0, 1)
            child1 = mutate(child1)
            child2 = mutate(child2)
            next_gen_weights.append(child1)
            next_gen_weights.append(child2)

        for i in range(total_models):
            current_pool[i].set_weights(next_gen_weights[i])
    else:
        print("Starting from Scratch")
    

def crossover(parent1, parent2):
    global current_pool
    parent1_weights = current_pool[parent1].get_weights()
    parent2_weights = current_pool[parent2].get_weights()
    child1_weights = copy.deepcopy(parent1_weights)
    child2_weights = copy.deepcopy(parent2_weights)

    gene = random.randrange(len(child1_weights))

    child1_weights[gene] = parent2_weights[gene]
    child2_weights[gene] = parent1_weights[gene]

    return child1_weights, child2_weights

def mutate(weights):
    model_weights = weights
    for i in range(len(model_weights)):
        for j in range(len(model_weights[i])):
            if random.random() < mutateProb:
                model_weights[i][j] += random.uniform(-0.5, 0.5)
    return model_weights

def select_action(state, modelNum):
    state = np.reshape(state, (1, input_dim))
    prob = current_pool[modelNum](state)[0]
    return np.argmax(prob)

initialize_models()

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


try:
    xrange
except NameError:
    xrange = range


def main():
    global SCREEN, FPSCLOCK, HITMASKS, ITERATIONS, VERBOSE, gen
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # # sounds
    # if 'win' in sys.platform:
    #     soundExt = '.wav'
    # else:
    #     soundExt = '.ogg'

    # SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    # SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    # SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    # SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    # SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    for gen in range(total_gen):
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.flip(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        with open(os.path.join(folder, "hitmasks_data.pkl"), "rb") as f:
            HITMASKS = pickle.load(f)

        print("Generation {0}: ".format(gen), end="")
        movementInfo = showWelcomeAnimation()
        crashInfo = mainGame(movementInfo)
        showGameOverScreen(crashInfo)
        print(crashInfo['score'])
        if save_current_pool:
            save_pool()



def showWelcomeAnimation():
    return {
            'playery': int((SCREENHEIGHT - PLAYER[IM_HEIGHT]) / 2),
            'basex': 0,
            'playerIndexGen': cycle([0, 1, 2, 1]),
            }


def mainGame(movementInfo):
    global fitness_pool
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerxList = [int(SCREENWIDTH * 0.2) for i in range(total_models)]
    playeryList = [movementInfo['playery'] for i in range(total_models)]


    basex = movementInfo['basex']
    baseShift = BASE[IM_WIDTH] - BACKGROUND[IM_WIDTH]

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH, 'y': newPipe1[0]['y'], 'actual_y': newPipe1[0]['actual_y']},
        {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y'], 'actual_y': newPipe2[0]['actual_y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY    =  [-9 for i in range(total_models)]   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward accleration
    playerRot     =  45   # player's rotation
    playerVelRot  =   3   # angular speed
    playerRotThr  =  20   # rotation threshold
    playerFlapAcc =  -9   # players speed on flapping
    playerFlapped = [False for i in range(total_models)] # True when player flaps
    # playersState = [None for i in range(total_models)]
    isPlayerAlive = [True for i in range(total_models)]

    total_alive = total_models

    while True:

        for i in range(total_models):
            if not isPlayerAlive[i]:
                continue
            fitness_pool[i] += 1
            state = np.array([  playerVelY[i],
                                BASEY - playeryList[i] - PLAYER[IM_HEIGHT],
                                lowerPipes[-2]['x'] - playerxList[i], 
                                lowerPipes[-2]['y'] - playeryList[i] - PLAYER[IM_HEIGHT],
                                lowerPipes[-1]['y'] - playeryList[i] - PLAYER[IM_HEIGHT]], dtype=np.float32)
            action = select_action(state, i)

            if action == 1:
                if playeryList[i] > -2 * PLAYER[IM_HEIGHT]:
                    playerVelY[i] = playerFlapAcc
                    playerFlapped[i] = True
            isPlayerAlive[i] = not checkCrash({'x': playerxList[i], 'y': playeryList[i], 'index': playerIndex}, upperPipes, lowerPipes)

            
        total_alive = sum(isPlayerAlive)
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

        # check for crash here
        if total_alive == 0:
            return {
                'y': 0,
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': 0,
                'playerRot': 0,
            }

        # check for score
        # playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        crossed_pipe = False
        for i in range(total_models):
            if not isPlayerAlive:
                continue
            for pipe in upperPipes:
                pipeMidPos = pipe["x"] + PIPE[IM_WIDTH] / 2
                if pipeMidPos <=  playerxList[i] +  PLAYER[IM_WIDTH] / 2 < pipeMidPos + 4:
                    crossed_pipe = True
                    fitness_pool[i] += 25
                
        if crossed_pipe:
            score += 1
                

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # # rotate the player
        # if playerRot > -90:
        #     playerRot -= playerVelRot

        playerHeight = PLAYER[IM_HEIGHT]
        # player's movement
        for i in range(total_models):

            if playerVelY[i] < playerMaxVelY and not playerFlapped[i]:
                playerVelY[i] += playerAccY
            if playerFlapped[i]:
                playerFlapped[i] = False

            # # more rotation to cover the threshold (calculated in visible rotation)
            # playerRot = 45
            
            playeryList[i] += min(playerVelY[i], BASEY - playeryList[i] - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -PIPE[IM_WIDTH]:
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)

        # Player rotation has a threshold
        # visibleRot = playerRotThr
        # if playerRot <= playerRotThr:
        #     visibleRot = playerRot
        
        # playerSurface = pygame.transform.rotate(IMAGES['player'][playerIndex], visibleRot)
        for i in range(total_models):
            if isPlayerAlive[i] == True:
                SCREEN.blit(IMAGES['player'][playerIndex], (playerxList[i], playeryList[i]))

        pygame.display.update()
        FPSCLOCK.tick(FPS)


def showGameOverScreen(crashInfo):

    global fitness_pool, current_pool, best_fitness, gen, best_model_weights

    isNewBestModel = False
    if best_model is not None:
        return
    total_fitness = np.sum(fitness_pool)
    next_gen_weights = []
    parent1, parent2 = fitness_pool.argsort()[-2:][::-1]

    if fitness_pool[parent1] >= best_fitness:
        isNewBestModel = True
        best_fitness = fitness_pool[parent1]
        best_model_weights = current_pool[parent1].get_weights()
        current_pool[parent1].save("best_model.h5")

    
    for i in range(total_models // 2):
        child1, child2 = crossover(parent1, parent2)
        if not isNewBestModel:
            child1 = best_model_weights

        child1 = mutate(child1)
        child2 = mutate(child2)
        next_gen_weights.append(child1)
        next_gen_weights.append(child2)
    
    for i in range(len(next_gen_weights)):
        fitness_pool[i] = 0
        current_pool[i].set_weights(next_gen_weights[i])
    

    """crashes the player down ans shows gameover image"""
    # score = crashInfo['score']
    # playerx = SCREENWIDTH * 0.2
    # playery = crashInfo['y']
    # playerHeight = IMAGES['player'][0].get_height()
    # playerVelY = crashInfo['playerVelY']
    # playerAccY = 2
    # playerRot = crashInfo['playerRot']
    # playerVelRot = 7

    # basex = crashInfo['basex']

    # upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

    


    # while True:
    #     for event in pygame.event.get():
    #         if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
    #             pygame.quit()
    #             sys.exit()
    #         if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
    #             if playery + playerHeight >= BASEY - 1:
    #                 return

    #     # player y shift
    #     # if playery + playerHeight < BASEY - 1:
    #     #     playery += min(playerVelY, BASEY - playery - playerHeight)

    #     # # player velocity change
    #     # if playerVelY < 15:
    #     #     playerVelY += playerAccY

    #     # # rotate only when it's a pipe crash
    #     # if not crashInfo['groundCrash']:
    #     #     if playerRot > -90:
    #     #         playerRot -= playerVelRot

    #     # draw sprites
    #     SCREEN.blit(IMAGES['background'], (0,0))

    #     for uPipe, lPipe in zip(upperPipes, lowerPipes):
    #         SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
    #         SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

    #     SCREEN.blit(IMAGES['base'], (basex, BASEY))
    #     showScore(score)

        


    #     playerSurface = pygame.transform.rotate(IMAGES['player'][1], playerRot)
    #     SCREEN.blit(playerSurface, (playerx,playery))
    #     SCREEN.blit(IMAGES['gameover'], (50, 180))

    #     FPSCLOCK.tick(FPS)
    #     pygame.display.update()


def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.57 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = PIPE[IM_HEIGHT]
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight, 'actual_y': gapY},  # upper pipe
        {"x": pipeX, "y": gapY + PIPEGAPSIZE},  # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player["index"]
    player["w"] = PLAYER[IM_WIDTH]
    player["h"] = PLAYER[IM_HEIGHT]

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:

        playerRect = pygame.Rect(player["x"], player["y"], player["w"], player["h"])
        pipeW = PIPE[IM_WIDTH]
        pipeH = PIPE[IM_HEIGHT]

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True

    return False

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()

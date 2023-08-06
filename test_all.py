import pytest
import pygame
import main as main
import time
import ctypes

# # use pygame to display the networks? Decreases speed
# visualize = True
#
# def init_pygame():
#     # allows resolution of pygame to be accurate
#     ctypes.windll.user32.SetProcessDPIAware()
#     # Technical
#     SCREEN_X = 3840 / 2
#     SCREEN_Y = 2160 / 2
#     FPS = 2000
#
#     # Aesthetic
#     BACKGROUND = (0, 0, 0)
#     LINE_WIDTH = 3
#     DEFAULT_LINE_COLOR = (255, 255, 255)
#     DEFAULT_NEURON_COLOR = (200, 255, 200)
#     CIRCLE_SIZE = 50
#     BUFFER = 200
#     # tries to keep min_spacing for all neurons
#     MIN_SPACING = 140
#     NEURON_COLOR_LIST = [(255, 0, 0), (50, 50, 50), (0, 255, 0)]
#     LINE_COLOR_LIST = [(255, 0, 0), (50, 50, 50), (0, 255, 0)]
#     DEFAULT_TEXT_COLOR = (255, 255, 255)
#
#     # Create Screen
#     screen = pygame.display.set_mode((SCREEN_X, SCREEN_Y), pygame.HWSURFACE | pygame.DOUBLEBUF)
#     pygame.display.set_caption('SLFS')
#     pygame.font.init()
#     font = pygame.font.Font('freesansbold.ttf', 32)
#
#     running = True
#     # FPS helper
#     start = 0
#     end = 0
#     i = 0
#
#
#     while running:
#         start = time.time()
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#
#         i += 0.01
#         print(i)
#         if i >= 1.0:
#             i -= 1
#             # print what the FPS max COULD be
#             text = font.render('FPP: ' + str(round(fps)), True, (255, 255, 255), BACKGROUND)
#             textRect = text.get_rect()
#             textRect.center = (SCREEN_X - 100, SCREEN_Y - 50)
#
#         # FPS Adjust
#         end = time.time()
#         diff = end - start
#         if diff != 0:
#             fps = 1 / diff
#         waitTime = 1 / FPS - diff
#
#
#
#         if waitTime > 0:
#             time.sleep(waitTime)
#         else:
#             pygame.draw.line(screen, (255, 0, 0), (SCREEN_X - 100, SCREEN_Y - 100), (SCREEN_X, SCREEN_Y), 10)
#         pygame.display.flip()
#
# if visualize:
#     init_pygame()


def test_getScreen():
    assert 1 == 1

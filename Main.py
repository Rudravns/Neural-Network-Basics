import sys
import os
import pygame
import Neural_Network as network
import helper
import numpy as np



class Window:
    def __init__(self, learning = False, test_acc = False, times = 1):
        pygame.init()
        self.screen = pygame.display.set_mode((784, 784), pygame.RESIZABLE)
        pygame.display.set_caption('Neural Network Display')
        self.clock = pygame.time.Clock()
        self.tick = 60


        #config
        training_config = {
            "input_size": 784,          # 28x28 pixels flattened
            "output_size": 11,          # digits 0-9
            "hidden_layers": (128, 100, 40),  # larger hidden layers for better learning
            "epochs": 10,               # enough to learn patterns without taking too long
            "batch_size": 32,           # smaller batch size for smoother updates
            "learning_rate": 0.1       # slightly higher for faster learning on small network
        }
        

        grid_config = {
            "size": 28,
            "rows": 28,
            "cols": 28
        }

        

        #grid data 
        self.size = grid_config["size"]
        self.rows = grid_config["rows"]
        self.cols = grid_config["cols"]
        w,h = self.screen.get_size()
        self.grid_data = np.zeros((h//self.size, w//self.size))

        

        #nnw
        if learning and os.path.exists(helper.MODEL_PATH):
            os.remove(helper.MODEL_PATH)
            print("Old model removed to start fresh training.")

        self.nw = network.Neural_Network(
            training_config["input_size"],
            training_config["output_size"],
            training_config["hidden_layers"]
        )

        self.trainer = network.Neural_Network_Trainer(self.nw)
        for i in range(times):
            if learning:
                self.trainer.train(
                    training_config["epochs"],
                    training_config["batch_size"],
                    training_config["learning_rate"]
                )
            if test_acc:
                self.trainer.test()
    
    def run(self):
        
        while True:
            self.screen.fill((0, 0, 0))
            self.clock.tick(self.tick)
            output = self.nw.forward(self.grid_data.flatten())
            predicted_num = np.argmax(output)
            formated_output = [f"{i}: {round(x*100,2)}%, " for i, x in enumerate(output)]
            
            
            
            # Handle drawing input (Brush logic)
            if pygame.mouse.get_pressed()[0] or pygame.mouse.get_pressed()[2]:
                mx, my = pygame.mouse.get_pos()
                col = mx // self.size
                row = my // self.size
                val = 1 if pygame.mouse.get_pressed()[0] else 0
                
                # Draw 1x1 brush for precision
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    self.grid_data[row, col] = val

            #self.draw_nodes()
            self.draw_grid()

            #text drawing
            self.draw_text(f"FPS: {round(self.clock.get_fps())}", 20, 20)
            self.draw_text(f"Predicted: {predicted_num}", 20, 50, size=40, color=(0, 255, 0))
            self.draw_text("Keys: 0-9 to teach, X for 10, S to Save", 20, 750, size=20)
            
            for i in range(11):
                color = (0, 255, 0) if i == predicted_num else (255, 255, 255)
                self.draw_text(f"Output: {formated_output[i]}", 20, 100+(i*40), size=20, color=color)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.VIDEORESIZE:
                    # The display is already updated by pygame in resizable mode,
                    # but we keep this to handle state if needed.
                    pass
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_c:
                        self.grid_data.fill(0)
                    
                    # Manual Teaching
                    if pygame.K_0 <= event.key <= pygame.K_9:
                        self.teach(event.key - pygame.K_0)
                    if event.key == pygame.K_x:
                        self.teach(10)
                    if event.key == pygame.K_s:
                        helper.save({"weights": self.nw.weights, "biases": self.nw.biases})
                        print("Model saved.")
            pygame.display.update()

    def teach(self, label):
        # Convert 0-1 grid to 0-255 for the trainer
        self.trainer.train_manual(self.grid_data * 255, label)
        print(f" taught {label}")

    def draw_text(self, Text, x, y, size = 30, color = (255, 255, 255)):
        font = pygame.font.SysFont('Arial', size)
        text = font.render(Text, True, color)
        self.screen.blit(text, (x,y))

    def draw_grid(self):
        w,h = self.screen.get_size()
        size = self.size
        rows, cols = self.grid_data.shape

        for x in range(cols):
            for y in range(rows):
                if self.grid_data[y,x] == 1:
                    pygame.draw.rect(self.screen, (255,255,255), (x*size,y*size,size,size))

    def draw_nodes(self):
        w, h = self.screen.get_size()
        sizes = self.nw.get_all_nodes()

        # Scale node radius and spacing based on window size
        node_radius = min(w, h) * 0.02
        LEFT_MARGIN = w * 0.1
        RIGHT_MARGIN = w * 0.9

        num_layers = 2 + len(sizes[1])
        layer_spacing = (RIGHT_MARGIN - LEFT_MARGIN) / (num_layers - 1)

        # ---- Input layer ----
        x = LEFT_MARGIN
        y_spacing = h / (sizes[0] + 1)

        for i in range(sizes[0]):
            pygame.draw.circle(
                self.screen, (0, 0, 255),
                (int(x), int(y_spacing * (i + 1))), int(node_radius)
            )

        # ---- Hidden layers ----
        for layer_index, layer_size in enumerate(sizes[1]):
            x = LEFT_MARGIN + layer_spacing * (layer_index + 1)
            y_spacing = h / (layer_size + 1)

            for i in range(layer_size):
                pygame.draw.circle(
                    self.screen, (255, 0, 0),
                    (int(x), int(y_spacing * (i + 1))), int(node_radius)
                )

        # ---- Output layer ----
        x = LEFT_MARGIN + layer_spacing * (num_layers - 1)
        y_spacing = h / (sizes[2] + 1)

        for i in range(sizes[2]):
            pygame.draw.circle(
                self.screen, (0, 255, 0),
                (int(x), int(y_spacing * (i + 1))), int(node_radius)
            )


if __name__ == '__main__':
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    os.system('cls') if os.name == 'nt' else os.system('clear')
 
    window = Window(learning=False, test_acc=False, times=1)
    window.run()
 

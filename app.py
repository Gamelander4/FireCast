from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Rectangle, Color, Line
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import NumericProperty
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

#data reading and preparing
np.set_printoptions(precision=3, suppress=True)
data = pd.read_csv("forest_fires.csv", header=0, names=["month", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "class"])
data_train = data.copy()
y = data_train.pop('class')
X = np.array(data_train)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

#data normalization
normal = tf.keras.layers.Normalization()
normal.adapt(X_train)

#keras sequential model creation
model = tf.keras.models.Sequential([
    normal,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss = tf.keras.losses.BinaryFocalCrossentropy(),
    metrics= tf.keras.metrics.BinaryAccuracy(),
)
model.fit(X_train, y_train, epochs=200)

class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout1 = BoxLayout(orientation='vertical')
        title = Label(text="FireCast", color=(1,1,0,1), font_size=60)
        with title.canvas.before:
            Color(1,0.5,0,1)
            self.rect = Rectangle(pos=title.pos, size=title.size)
        title.bind(pos=self.update_rect, size=self.update_rect)
        layout1.add_widget(title)
        layout2 = BoxLayout(orientation="horizontal")
        pred_screen = Button(text="Predict Wildfire!", on_release=self.switch_to_predict, color=(1,1,0,1))
        pred_screen.background_color=(1,0,0,1)
        about_screen = Button(text="About Me!", on_release=self.switch_to_me, color=(1,1,0,1))
        about_screen.background_color=(1,0,0,1)
        layout2.add_widget(pred_screen)
        layout2.add_widget(about_screen)
        layout1.add_widget(layout2)
        self.add_widget(layout1)
    def switch_to_predict(self, instance):
        self.manager.current = 'PredictPage'
    def switch_to_me(self, instance):
        self.manager.current = 'AboutMe'
    def update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size
class BorderLabel(Label):
    border_width = NumericProperty(2)

    def __init__(self, **kwargs):
        super(BorderLabel, self).__init__(**kwargs)
        with self.canvas.before:
            Color(1, 0.5, 0, 1) 
            self.border = Line(rectangle=(self.x, self.y, self.width, self.height), width=self.border_width)
    def on_pos(self, instance, value):
        self.border.rectangle = (self.x, self.y, self.width, self.height)

    def on_size(self, instance, value):
        self.border.rectangle = (self.x, self.y, self.width, self.height)
class PredictPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hori_whole = 1

        layout1 = BoxLayout(orientation='vertical')
        layout2 = BoxLayout(orientation='horizontal', size_hint=(1,0.1))
        field_input = BoxLayout(orientation='horizontal', size_hint=(1,0.4))
        side1 = BoxLayout(orientation='vertical',spacing=4)
        side2 = BoxLayout(orientation='vertical')
        pred_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.2))
        intro_layout = BoxLayout(orientation='vertical', size_hint=(1,0.3))
        
        title = Label(text="FireCast", font_size=30, color=(1,1,0,1), size_hint=(hori_whole, hori_whole))
        title.bind(pos=self.red_update_rect, size=self.red_update_rect)
        warn = Label(text="Fill out as much information as you can for a more accurate prediction.\nFFMC, DMC, DC, and ISI are all indexes that relate to predicting fires.", color=(1,1,0), size_hint=(hori_whole, hori_whole))
        warn.bind(pos=self.orange_update_rect, size=self.orange_update_rect)
        
       
        self.text_input1 = TextInput(size_hint=(hori_whole, hori_whole))
        self.text_input2 = TextInput(size_hint=(hori_whole, hori_whole))
        self.text_input3 = TextInput(size_hint=(hori_whole, hori_whole))
        self.text_input4 = TextInput(size_hint=(hori_whole, hori_whole))
        self.text_input5 = TextInput(size_hint=(hori_whole, hori_whole))
        self.text_input6 = TextInput(size_hint=(hori_whole, hori_whole))
        self.text_input7 = TextInput(size_hint=(hori_whole, hori_whole))
        self.text_input8 = TextInput(size_hint=(hori_whole, hori_whole))

        label1 = BorderLabel(text='Month (MM):', size_hint=(hori_whole, hori_whole), color=(0,0,0,1))
        label1.bind(pos=self.white_update_rect, size=self.white_update_rect)
        label2 = BorderLabel(text='Fine Fuel Moisture Code (FFMC):', size_hint=(hori_whole, hori_whole), color=(0,0,0,1))
        label2.bind(pos=self.white_update_rect, size=self.white_update_rect)
        label3 = BorderLabel(text='Duff Moisture Code (DMC):', size_hint=(hori_whole, hori_whole), color=(0,0,0,1))
        label3.bind(pos=self.white_update_rect, size=self.white_update_rect)
        label4 = BorderLabel(text='Drought Code (DC):', size_hint=(hori_whole, hori_whole), color=(0,0,0,1))
        label4.bind(pos=self.white_update_rect, size=self.white_update_rect)
        label5 = BorderLabel(text='Initial Spread Index (ISI):', size_hint=(hori_whole, hori_whole), color=(0,0,0,1))
        label5.bind(pos=self.white_update_rect, size=self.white_update_rect)
        label6 = BorderLabel(text='Temperature in Celsius:', size_hint=(hori_whole, hori_whole), color=(0,0,0,1))
        label6.bind(pos=self.white_update_rect, size=self.white_update_rect)
        label7 = BorderLabel(text='Relative Humidity Percentage between 1 and 100:', size_hint=(hori_whole, hori_whole), color=(0,0,0,1))
        label7.bind(pos=self.white_update_rect, size=self.white_update_rect)
        label8 = BorderLabel(text='Wind Speed in km/h:', size_hint=(hori_whole, hori_whole), color=(0,0,0,1))
        label8.bind(pos=self.white_update_rect, size=self.white_update_rect)

        pred_button = Button(text="Predict Wildfire!", on_release=self.predicting, size_hint=(0.4, 0.5), color=(1, 1, 0, 1), font_size=25, pos_hint={'x':0.3})
        pred_button.background_color=(0.8,0,0,1)
        self.prediction_label = Label(size_hint=(1, 0.5), font_size=25, color=(0, 0, 0, 1))
        self.prediction_label.bind(pos=self.white_update_rect, size=self.white_update_rect)
        home = Button(text="Home!", on_release=self.switch_to_home, size_hint=(hori_whole, hori_whole), color=(1, 1, 0, 1))
        home.background_color=(1,0,0,1)
        about = Button(text="About Me!", on_release=self.switch_to_me, size_hint=(hori_whole, hori_whole), color=(1, 1, 0, 1))
        about.background_color=(1,0,0,1)

        side1.add_widget(label1)
        side1.add_widget(label2)
        side1.add_widget(label3)
        side1.add_widget(label4)
        side1.add_widget(label5)
        side1.add_widget(label6)
        side1.add_widget(label7)
        side1.add_widget(label8)

        side2.add_widget(self.text_input1)
        side2.add_widget(self.text_input2)
        side2.add_widget(self.text_input3)
        side2.add_widget(self.text_input4)
        side2.add_widget(self.text_input5)
        side2.add_widget(self.text_input6)
        side2.add_widget(self.text_input7)
        side2.add_widget(self.text_input8)

        intro_layout.add_widget(title)
        intro_layout.add_widget(warn)
        field_input.add_widget(side1)
        field_input.add_widget(side2)
        pred_layout.add_widget(pred_button)
        pred_layout.add_widget(self.prediction_label)
        layout2.add_widget(home)
        layout2.add_widget(about)
        layout1.add_widget(intro_layout)
        layout1.add_widget(field_input)
        layout1.add_widget(pred_layout)
        layout1.add_widget(layout2)
        self.add_widget(layout1)
    def predicting(self, instance):
        text1_value = self.text_input1.text; text2_value = self.text_input2.text; text3_value = self.text_input3.text
        text4_value = self.text_input4.text; text5_value = self.text_input5.text; text6_value = self.text_input6.text
        text7_value = self.text_input7.text; text8_value = self.text_input8.text
        list = []
        all_vals = [text1_value, text2_value, text3_value, text4_value, text5_value, text6_value, text7_value, text8_value]
        for i in all_vals:
            if i != '':
                try:
                    flo_form = float(i)
                    list.append(flo_form)
                except Exception as e:
                    self.prediction_label.text = "Please enter in numbers"
                    return
            else:
                list.append(-1.0) 
        input = np.array(list)
        input_array = input.reshape(1, 8)
        self.predict = model.predict(input_array)
        self.predict = self.predict[0][0] * 100
        self.predict = f"{self.predict:.2f}" 
        self.prediction_label.text = f"{self.predict}% chance of a wildfire occuring!"
    def switch_to_home(self, instance):
        self.manager.current = 'home'
    def switch_to_me(self, instance):
        self.manager.current = 'AboutMe'
    def red_update_rect(self, instance, value):
        instance.canvas.before.clear()  
        with instance.canvas.before:
            Color(0.8, 0, 0, 1) 
            self.rect = Rectangle(pos=instance.pos, size=instance.size)
    def orange_update_rect(self, instance, value):
        instance.canvas.before.clear()  
        with instance.canvas.before:
            Color(1, 0.5, 0, 1)  
            self.rect = Rectangle(pos=instance.pos, size=instance.size)
    def white_update_rect(self, instance, value):
        instance.canvas.before.clear() 
        with instance.canvas.before:
            Color(1, 1, 1, 1)  
            self.rect = Rectangle(pos=instance.pos, size=instance.size)
class Me(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        title = Label(text="About Me", color=(1,1,0,1), font_size=60)
        title.bind(pos=self.red_update_rect, size=self.red_update_rect)
        about = Label(text="My name is Praneel Nemani. \nI am a high school junior and a Boy Scout passionate about \nusing technology to make a difference in the community.", color=(0,0,0,1), font_size=30)
        about.bind(pos=self.white_update_rect, size=self.white_update_rect)
        layout2 = BoxLayout(orientation='horizontal')
        home = Button(text="Home!", on_release=self.switch_to_home, color=(1,1,0,1))
        home.background_color=(1,0,0,1)
        predict = Button(text="Predict Wildfire!", on_release=self.switch_to_predict, color=(1,1,0,1))
        predict.background_color=(1,0,0,1)
        layout.add_widget(title)
        layout.add_widget(about)
        layout2.add_widget(home)
        layout2.add_widget(predict)
        layout.add_widget(layout2)
        self.add_widget(layout)

    def switch_to_home(self, instance):
        self.manager.current = 'home'

    def switch_to_predict(self, instance):
        self.manager.current = 'PredictPage'
    def red_update_rect(self, instance, value):
        instance.canvas.before.clear() 
        with instance.canvas.before:
            Color(0.8, 0, 0, 1) 
            self.rect = Rectangle(pos=instance.pos, size=instance.size)
    def white_update_rect(self, instance, value):
        instance.canvas.before.clear()
        with instance.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(pos=instance.pos, size=instance.size)
class MyApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(PredictPage(name='PredictPage'))
        sm.add_widget(Me(name='AboutMe'))
        return sm
if __name__ == '__main__':
    MyApp().run()
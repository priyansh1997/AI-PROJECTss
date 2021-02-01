# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:41:05 2020

@author: PRIYANSH facebook chatbot
"""

import random, os, sys, time, re
from selenium import webdriver
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd

browser = webdriver.Chrome('E:/AI/Facebook_automation/driver/chromedriver.exe')
browser.get('https://www.facebook.com')

file = open('E:/AI/Facebook_automation/config.txt')

lines = file.readlines()

username = lines[0]
pswd = lines[1]

elementID = browser.find_element_by_id('email')
elementID.send_keys(username)

elementID = browser.find_element_by_id('pass')
elementID.send_keys(pswd)

elementID.submit()

visiting_profile_id = '<profileid>'
full_link = 'https://www.facebook.com' + visiting_profile_id

browser.get(full_link)

visited_profile = []
profile_queued = []

#to get the source codeof page
#browser.page_source

##class and other enteries can be changed by the website owner 
def new_profile_ID(soup, profile_queued):
    profileID = []
    chat = soup.find_element_by_class_name('uiScrollableAreaContent')
    online_people = chat.find_elements_by_tag_name('a')
    for link in online_people:
        userID = link.get_attribute('href')
        if ((userID not in profile_queued) and (userID not in visited_profile)):
            profileID.append(userID)
    return profileID


profile_queued = new_profile_ID(browser, profile_queued)


import pyttsx3
import speech_recognition as sr
import pyaudio



#sapi5 to take voices 
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
#print(voices[1].id)
#voices[0] for david voice
engine.setProperty('voice',voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()



def takecmd():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print("listening...")
        #press control and click on command to view its working
        r.pause_threshold = 1
        audio=r.listen(source)
    
    
    try:
        print("Recognizing...")
        query=r.recognize_google(audio,language='en-in')
        print(f"user said: {query}\n")


    
    except Exception as e:
        #print(e)
        print("say that again please")
        #returning string
        return "None"
    return query

if __name__=="__main__":
    speak(k)
    #takecmd()


people_to_chat = browser.find_elements_by_xpath('//*[@id="u_0_2c"]/div/div/div[4]')
people_to_chat_more_contacts = browser.find_elements_by_xpath('//*[@id="js_sh"]/ul[2]')
total_on_chat = people_to_chat + people_to_chat_more_contacts


for i in total_on_chat:
    print(i.text)
    speak(i.text)
    

list_onchat_people_and_links = [on_chat, people_to_chat]
on_chat = pd.Series(on_chat)



















    
browser.find_element_by_link_text("Aniket Chaturvedi").click()
msg = query

classm = browser.find_element_by_class_name('_1d4_')
chat_class = classm.find_element_by_xpath('//*[@id="cch_f36acb99458eb68"]/div/div/div/div/div/div/div/span/span')
chat_class.send_keys(msg)
send_key_class = browser.find_element_by_xpath('//*[@id="js_4gs"]/div[4]/div/ul')


#we have to separately run this line to send the msg
send = send_key_class.find_element_by_class_name('svgIcon').click()







































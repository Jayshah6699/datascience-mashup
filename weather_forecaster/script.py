from selenium import webdriver

drive = webdriver.Chrome()

city = input("Enter city for which you want the weather forecast: ")

drive.get("https://www.weather-forecast.com/locations/"+city+"/forecasts/latest")

print(drive.find_element_by_class_name("b-forecast__table-description-content")[0].text)
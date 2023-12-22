# Cloud Computing Garden Capstone

API Documentation
* Use Python as programming language
* Use Flask as backend
* Use Cloud run for deployment
* Use Docker for containerization
* USe Postman for testing API

Plant Prediction API

Endpoint:
https://garden-capstone-2-le5664txwa-as.a.run.app/

Prediction:

* URL : /prediction

* Method : POST

* Request Body :

* Image : image_file.png/.jpg/.jpeg

* Respone : 200 OK

{

    "data": {
        "confidence": 19.0,
        "plant_types_prediction": "papaya"
    },
    "status": {
        "code": 200,
        "message": "Success predicting"
    }
}


![Postman](https://github.com/SmartGardenPlant/garden-capstone/assets/67750065/e1bcd68e-dffe-41d1-a4e1-f7ad1b6b8553)


   

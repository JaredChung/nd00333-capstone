import requests
import json

# URL for the web service, should be similar to:
# 'http://17fb482f-c5d8-4df7-b011-9b33e860c111.southcentralus.azurecontainer.io/score'
scoring_uri = 'http://af889be7-14d8-48e8-9205-0338fe6afdcb.southcentralus.azurecontainer.io/score'



# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "city": "city_103",
            "city_development_index": 0.92,
            "gender": 'Male',
            "relevent_experience": 'Has relevent experience',
            "enrolled_university": 'no_enrollment',
            "education_level": 'Graduate',
            "major_discipline": 'STEM',
            "experience": '>20',
            "company_size": '',
            "company_type": '',
            "last_new_job": "1",
            "training_hours": 36
          },
          {
            "city": "city_40",
            "city_development_index": 0.775,
            "gender": 'Male',
            "relevent_experience": 'No relevent experience',
            "enrolled_university": 'no_enrollment',
            "education_level": 'Graduate',
            "major_discipline": 'STEM',
            "experience": '15',
            "company_size": '50-99',
            "company_type": 'Pvt Ltd',
            "last_new_job": ">4",
            "training_hours": 47
          },

    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())



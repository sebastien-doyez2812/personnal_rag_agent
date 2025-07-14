
# 🤖💻 Create an Advanced Rag Agent with Langgraph and Qdrant 💻🤖

This repo is about an advanced rag agent.
The architecture is a backend with fastAPI ( I used REST API for this project), and the front end is created with React.
I used ollama to run locally LLMS, feel free to use OpenAI API if you have one.
I used Qwen2.5 as the LLM.
You can also have access to the metrics with langfuse ( don't forget to create a .env for your APIs).

For this project, I used Langgraph, Qdrant as the VectorDB. This agent is advanced because It shows images from the image collection (Qdrant) and give you answer using context store in the VectorDatabase. But if there are nothing on the topic asked in the vectorDB, the agent with search on the Web!
## Authors

- [@seb_doyez](https://github.com/sebastien-doyez2812)


## Installation
### Requirements:

* Clone this project:
 `git clone + url of my project`
* Run the backend:
`cd backend`


`uvicorn main:app --reload`
* Run the frontend:

`cd frontend`


`npm run dev`

## 📹 Youtue demo:

I made a quick demo on this project:
[click here for the video](https://www.youtube.com/watch?v=kwCLdQqM8bc)


There are also a quick explanation about the architercture and the agent workflow.
## Feedback

If you have any feedback, please reach out to us on Linkedin: 
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](
https://www.linkedin.com/in/s%C3%A9bastien-doyez-042604252/)


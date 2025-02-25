pipeline {
    agent any

    stages {
        stage('Build and run the mlflow docker containers'){
            steps{
                script{
                    bat 'docker compose up --build -d'
                }
            }
        }
    }

    post {
        success {
            echo "MLflow pipeline and model deployment done successfully!"
        }
        failure {
            echo "There was an error in running and model deployment..."
        }
    }
}
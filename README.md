# MLOps

MLOps project for showcasing some practices and tools for managing machine learning models.

This project has been created as part of the Software Engineering for AI Systems course at the University of Rostock. The course gave each student the opportunity to create a presentation on a topic of their choice. My presentation was about MLOps, and this repository was created to showcase some practices and tools for managing machine learning models.

# Tools and Practices

## Tools

- [DVC](https://dvc.org/) (Data Version Control)
- [MLflow](https://mlflow.org/) (Model Tracking)
- [CML](https://cml.dev/) (CI/CD for ML)
- [GitHub Actions](https://github.com/features/actions) (CI/CD)
- [DagsHub](https://dagshub.com/) (GitHub for Data Science)

## Practices

### Data Versioning

- Use DVC to version data and models.

### Model Versioning

- Use DVC to version models.

### Model Tracking

- Use MLflow to track models and experiments (metrics, parameters, artifacts).

### Continuous Integration

- Use GitHub Actions to automate the CI/CD pipeline.

### PR Review

- Use CML to run tests and generate reports for PRs.

### CD4ML

1. identify and prepare the data for training
2. experiment with different models to find best performing candidate
3. deploy and use selected model in production

This Project focuses on the first two steps of the CD4ML process. The data used is the well-known MNIST dataset, and the model used is a simple feedforward neural network. In order to show the second step, two different models are trained and compared using MLflow. The whole process is automated using DVC, CML, and GitHub Actions.

# Project Structure

The project structure is as follows (including only the relevant files and directories for MLOps):

```
.
├── data # data directory for dvc
├── metrics # metrics directory for mlflow
├── .github # github actions directory
│   └── workflows
│       └── cml.yaml # cml workflow (CI)
├── .dvc # dvc config directory
|
├── dvc.yaml # dvc config file
├── train.py # training script executed directly
├── train_parallel.py

```

# How to Use

Currently this repository is set up to run the training script using GitHub Action Secrets as environment variables.
This is due to showing the use of GitHub Actions and CML for PR review. Also a good practice for mlops is to run the training on external resources like a cloud provider which is (in a minimal way) simulated here using GitHub Actions Resources.

To run the training script yourself and see all elements in action, follow these steps:

1. Fork this repository.
2. Create a linked project on DagsHub for the same repository.
3. Go to the `Settings` tab of your fork.
4. Go to the `Secrets` section.
5. Add the following secrets:

- `DAGSHUB_ACCESS_TOKEN`: Access token for DagsHub.
- `MLFLOW_TRACKING_PASSWORD`: Password for MLflow tracking server (on DagsHub).
- `GIT_EMAIL`: Email for git (used for pushing changes to model and metrics meta files automatically).
- `GIT_NAME`: Name for git (used for pushing changes to model and metrics meta files automatically).
- `REPO_TOKEN`: Token for GitHub Actions to push changes to the repository.

After adding the secrets, you can create a new branch, make some changes, and open a PR to see the CI in action.

To showcase the PR review process an example PR has been created [here](https://github.com/NoahLeu/mlops-presentation/pull/38)

# Contributors

- [Noah Leu](https://github.com/NoahLeu)

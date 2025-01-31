# Create an Agent

In this notebook, an Amazon Bedrock Agent is created that connects to a single Knowledge Base for Amazon Bedrock to retrieve data and complete tasks.

For demonstration purposes we will use a pdf containing information about all the model offerings in AWS Bedrock.

The steps to complete this notebook are:

1. Import the needed libraries
2. Create an S3 bucket and upload the data to it
3. Create the Knowledge Base for Amazon Bedrock and sync data to the Knowledge Base
4. Create the Agent for Amazon Bedrock
5. Test the Agent
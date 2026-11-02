# AWS IAM Identity Center

AWS IAM Identity Center (successor to AWS SSO) as a SAML 2.0 identity provider for a customer-managed application.
The examples use Starlight at `https://starlight.example.com` and the slug `aws`.

## Set up the application

1. In the IAM Identity Center console, add a customer-managed application and choose SAML 2.0.
2. Download the IAM Identity Center SAML metadata file; you will paste it into Starlight.
3. Set the application ACS URL to `https://starlight.example.com/auth/aws/saml/acs` and the application SAML audience to
   `https://starlight.example.com/auth/aws/saml/metadata`.
4. Edit the attribute mappings: map the subject to a stable user identifier with the persistent format, and add
   attributes named `email` and `name` from the user's email and name.
5. Assign the users and groups who may use Starlight.

| Starlight field | Value |
| --- | --- |
| IdP metadata XML | The file from step 2 |
| Email domains | The domains your people sign in with |

## Groups

Control who can sign in by assigning users and groups to the application in IAM Identity Center. If your directory
exposes group membership as an application attribute, map it to an attribute named `groups` and Starlight syncs it;
otherwise manage project membership in Starlight.

## Check it works

Open Starlight from the AWS access portal or from `https://starlight.example.com/auth/login`. Starting from the access portal is
IdP-initiated sign-in, which Starlight refuses unless the connection allows it; starting from Starlight always works.

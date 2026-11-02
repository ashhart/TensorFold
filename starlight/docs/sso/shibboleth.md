# Shibboleth IdP

A Shibboleth Identity Provider over SAML 2.0, common in universities and research organisations. Starlight reads the
eduPerson and X.500 attributes by their standard names. The examples use Starlight at `https://starlight.example.com` and the
slug `shib`.

## Register Starlight

1. Add Starlight's SP metadata, `https://starlight.example.com/auth/shib/saml/metadata`, as a metadata provider on the IdP.
2. Release these attributes to Starlight in the attribute filter:

| Attribute | Name Starlight reads |
| --- | --- |
| mail | `urn:oid:0.9.2342.19200300.100.1.3` |
| displayName | `urn:oid:2.16.840.1.113730.3.1.241` |
| eduPersonPrincipalName | `urn:oid:1.3.6.1.4.1.5923.1.1.1.6` |
| isMemberOf | `urn:oid:1.3.6.1.4.1.5923.1.5.1.1` |

3. Issue a persistent NameID for Starlight, or release a stable identifier such as `subject-id` and enter its name as
   Starlight's subject attribute.
4. Give Starlight the IdP's metadata URL or XML.

Transient NameIDs change at every sign-in and cannot identify an account, so Starlight refuses them unless a subject
attribute is configured.

## Logout

Starlight's metadata lists its single logout endpoint and the certificate it signs logout messages with. Turn on **Sign out
of the IdP too** in Starlight to send a LogoutRequest to the IdP at sign-out. For a logout that starts at the IdP to reach
Starlight, the IdP must track SP sessions (`idp.session.trackSPSessions`).

## Groups

`isMemberOf` carries group names, often from Grouper; Starlight syncs them at every sign-in.

## Check it works

Sign in from `https://starlight.example.com/auth/login`.

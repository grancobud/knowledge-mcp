Corporate Good Practice

Issue/Amendment
1.7

Page
1/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

Table of contents

Introduction .............................................................................................................. 2
Purpose ................................................................................................................... 2
Scope ...................................................................................................................... 2
Goal ......................................................................................................................... 2
Terms and Definitions .............................................................................................. 3
Roles and Responsibilities ...................................................................................... 3
Incident Reporting Entity ......................................................................................... 3
IT Owner .................................................................................................................. 3
Bosch CERT ............................................................................................................ 4
Incident Handling Officer ......................................................................................... 4
C/CY ........................................................................................................................ 4
C/CYG ..................................................................................................................... 4
Corporate Crisis Management Team....................................................................... 5
C/ISP ....................................................................................................................... 5
OE Cybersecurity Officer (OE CSO) ........................................................................ 5
Local Cybersecurity Contact (Local CSC) ............................................................... 5
Cybersecurity Incident Report ................................................................................. 6
Cybersecurity Incident Report Elements ................................................................. 6
Prioritization ............................................................................................................. 6
Attack Vectors ......................................................................................................... 7
Security Incident Classification ................................................................................ 7
Information and Cybersecurity Incident Severity ..................................................... 8
Information Sensitivity ........................................................................................... 10
Cybersecurity Incident Response Lifecycle ........................................................... 12
Prepare .................................................................................................................. 13
Detection, & Analysis ............................................................................................. 13
Containment, Eradication, and Recovery .............................................................. 14
Post-Incident Activity ............................................................................................. 14
Cybersecurity Response Activities ........................................................................ 15
Appendix ................................................................................................................ 20
Incident Reporting Form ........................................................................................ 20
Obligations ............................................................................................................. 20
Revision History ..................................................................................................... 20

1
1.1
1.2
1.3
1.4
2
2.1
2.2
2.3
2.4
2.5
2.6
2.7
2.8
2.9
2.10
3
3.1
3.2
3.3
3.4
3.5
3.6
4
4.1
4.2
4.3
4.4
5
6
6.1
6.2
7

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
2/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

1

Introduction

1.1  Purpose

Today, cyber attacks are imminent - this means cyber attacks on the Bosch group are no longer a
question of if, but when. Conventional technology-driven security such as preventive measures
are not sufficient anymore. Detection is equally as important, and Incident Response (IR)
becomes even more important in today’s cybersecurity threat landscape.

This document describes the overall plan for responding to cybersecurity incidents at Bosch. It
describes the incident response lifecycle and incident communication plan that defines interfaces,
roles and responsibilities, incident classification scheme, and security incident reporting
requirements. This document is revised once a year by C/CYG and C/CYT and altered if
necessary. The version history can be found at the end of the document.

1.2  Scope

This plan applies to all information or cybersecurity incidents occurring within the Bosch Group.
The detection, analysis, and response of security incidents require communication and
coordination with many entities and reliance on cooperation with the incident response approach.
This Cybersecurity Incident Response Plan (CIRP) is based on current procedures and
capabilities – this code of practice enables the Bosch Group to respond to cybersecurity incidents
at Bosch effectively.

This CIRP does not cover handling and communication aspects in case of Privacy related
incidents.

1.3  Goal

The goal of the CIRP is to define the responsibilities and the communication structure in case of
information or cybersecurity incidents:

  Prepare, detect, analyze, contain, eradicate, and help to recover from security incidents
  Respond to security incidents systematically so that the appropriate steps are taken
  Help to recover quickly and efficiently from security incidents, minimizing loss or theft of

information and disruption of services

  Regular and suitable communication during an incident and post incident reporting
  Using information gained during incident handling to better prepare for handling future

incident and to provide stronger protection for systems and data

  Mitigate cybersecurity risk by corrective actions

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
3/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

1.4  Terms and Definitions

The terms and definitions of CD 09000 apply.

2  Roles and Responsibilities

This chapter defines the roles and responsibilities for cybersecurity incident response:

2.1

Incident Reporting Entity

Cybersecurity incidents can be reported from any entity inside as well as outside the Bosch Group
and is considered as incident reporting entity.

2.2

IT Owner

The responsibilities of the IT Owner in the context of a cybersecurity incident are defined in CD
09000. Especially, Chapters “2.5 “Cybersecurity incident management planning and preparation
in Attachment 001 of CD 09000.

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
4/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

2.3  Bosch CERT

The roles and responsibilities of Bosch CERT are defined in Attachment 002 of CD 09000
Chapter 3 “Bosch CERT”.

Bosch CERT establishes the role Incident Handling Officer and leads the response to IT related
cybersecurity incidents.

2.4

Incident Handling Officer

An Incident Handling Officer is member of the Bosch CERT and must lead the response to
cybersecurity incidents. Tasks and responsibilities of the Incident Handling Officer are:

  Responsible for immediate notification of an incident according to section 5
  Determine whether the incident has the level of a cybersecurity crisis and escalate to

C/CY

  Responsible for making all tactical decisions regarding the incident
  Coordinate the development of the incident response strategy and then create and assign

response actions to supporting entities

  Determine initial incident stakeholders and initiate the incident notification process
  Determine if work package leads are needed and assign as necessary
  Lead initial incident notification and subsequent incident status updates according to

section 5

  Document corrective actions identified during the incident response process
  Determine when incident is officially resolved and closed
  Conduct lessons learned meeting and document the official Incident Report

In case of a cybersecurity crisis situation, the role of incident handling officer is covered within the
C/CY organization.

2.5  C/CY

The roles and responsibilities of C/CY are defined in RBGF 177.

Tasks of C/CY members in case of cybersecurity incidents considered as crisis (Definition of
crisis see RB/GF184):



Incident Handling Officer of Global Cybersecurity Incidents

2.6  C/CYG

The roles and responsibilities of C/CYG are defined in Attachment 002 of CD 09000 Chapter 2
“Cybersecurity Governance”.

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
5/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

2.7  Corporate Crisis Management Team

The Corporate Crisis Management Team (CCMT) coordinates and manages crises.
Details are provided in RB/GF184.

2.8  C/ISP

C/ISP must be informed by Bosch CERT in case of significant Information security incidents. .The
term “significance regarding Data Protection or Information Security” includes at least all topics
with Bosch corporate group wide strategic relevance or potential high risk. Additional information
are provided in Section 3.5 regarding the Information and Cybersecurity Incident Severity. In case
of doubts C/ISP should be contacted.

Information security incidents and the knowledge obtained from these, must be classified as
security class C-SC3 or I-SC3. The release of this information outside the Bosch Group must be
agreed with C/ISP and C/CYG in advance. (See C/ISP CD 02900 & CD 09000 A16.1.6)

* C/ISP must be informed in case of probable cause of privacy breach

* Responsible for conformity to regulations for public obligation to inform (details see RB/GF105

2.9  OE Cybersecurity Officer (OE CSO)

The roles and responsibilities of the OE CSO are defined in CD 09000 Chapter 4.2.2.:

  The OE CSO must support Bosch CERT regarding cybersecurity incidents where IT Systems

or Bosch products in the responsibility of the OE are affected.

2.10  Local Cybersecurity Contact (Local CSC)

The roles and responsibilities of the local CSC are defined in CD 09000 Chapter 4.2.5.:





(3) Must support C/CY, particularly Bosch CERT, in case of a cybersecurity incident in the
location,
(4) Has the right to execute instructions given by Bosch CERT (see Attachment 2) in case of a
cybersecurity incident, (e.g., disconnecting devices from the Bosch network), and the right to
physically access the respective areas.

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
6/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

3  Cybersecurity Incident Report

3.1  Cybersecurity Incident Report Elements

Bosch CERT has established a standard set of data elements to collect for each incident report.
Incident reporting entities may be asked to provide the following information when reporting a
security incident.

  Contact information for both the impacted and reporting organizations.
  Details describing any vulnerabilities involved (i.e., Common Vulnerabilities and

Exposures (CVE) identifiers)

  Date and time of occurrence, including time zone.
  Date and time of detection and identification, including time zone.
  Related indicators (e.g, hostnames, domain names, network traffic characteristics, registry

keys, X.509 certificates, MD5 file signatures)

  Attack vectors, if known (see section 3.3)
  Prioritization factors (see section 3.2)
  Source and Destination Internet Protocol (IP) address, port, and protocol.
  Asset information (see section 3.6)

  Virtual system environment(s)
  Virtual system location(s)
  System Function(s) (e.g, web server, domain controller, or workstation)
  Asset owner

  Mitigation actions taken, if applicable
  Sources, methods, or tools used to identify the incident (e.g., Intrusion Detection System

(IDS), Intrusion Prevention System (IPS) or audit log analysis)

3.2  Prioritization

Bosch CERT incident handling prioritization is based on the information security incident severity
as described in section 3.5.

A security breach can have severe business effects due to the loss of reputation and trust.

Priority rating can be (and often is) adjusted during the course of the information security
incident handling process.

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
7/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

3.3  Attack Vectors

The attack vector has to be determined (if possible) and documented for cybersecurity incidents
with at least medium severity. The attack vectors will be defined by the Incident Response Team,
examples can be found at NIST SP 800-61.

3.4  Security Incident Classification

Bosch incorporates the following incident classification taxonomy, which is based on the widely
adopted “eCSIRT.net mkVI” taxonomy, to help drive the incident response actions for incidents
occurring within the Bosch Group.

Incident category

Description

Example

Abusive Content

Malicious Code

Information Gathering

Intrusion Attempts

Intrusions

Availability

Inappropriate activities or
behaviour with a bad effect or for
a bad purpose.

Spam (unsolicited bulk email),
Scam, Harmful content,
Discrimination, Misinformation

Software that is intentionally
included or inserted in a system
for a harmful purpose. A user
interaction is normally necessary
to activate the code.

Information Gathering is the act
of gathering different kinds of
information against the targeted
victim or system.

An attempt to compromise a
system or to disrupt any service
by using different techniques.

A successful compromise of a
system or application (service).
This can have been caused
remotely by a known or new
vulnerability, but also by an
unauthorized local access.

By this kind of an attack a
system is bombarded with so
many packets that the operations
are delayed or the system
crashes.

Malware: Virus, Worm, Trojan,
Spyware, Dialler, Rootkit, etc.

Scanning, Footprinting,
Enumeration, Sniffing, Social
Engineering

Exploiting known or unknown
vulnerabilities, Login attempts

(Un-)Privileged account
compromise, Application
compromise, Botnet

DoS, DDos, Sabotage, Outage
(no malice)

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
8/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

Information Content Security

Unauthorised access to or
modification of information.

Fraud

Vulnerable

Other

Test

Fraud is intentional deception to
secure unfair or unlawful gain, or
to deprive a victim of a legal
right.

A weakness of an asset or group
of assets that can be exploited
by one or more threats.

All incidents which do not fit in
one of the given categories
should be put into this class.

Meant for testing.

Wiretapping, Spoofing, Hijacking,
Error caused by Human,
Configuration, or Software

Unauthorized use of resources,
Copyright, Masquerading
(phishing for identity theft)

Vulnerability in software

Testing Adversarial Behaviour,
Red Teaming

3.5

Information and Cybersecurity Incident Severity

Security incident severity ratings are used to prioritize Bosch CERT’s incident response actions.
Only security incidents are assigned severity ratings. Security alerts do not receive a severity
rating until they are confirmed security incidents. The following security incident severity matrix
prioritizes security incidents.

Information Security Incident Severity Matrix

Severity
Rating

Operational
damage

Reputational
damage

Knowhow loss

Examples

Severity 3
(Critical)

Operational
processes are
massively
disrupted (i.e.,
disruption of
manufacturing
processes).
Transition to
replacement
processes not
possible or only
possible through

Wide and large
lasting negative
effects that are
reported in the
media.

Loss of
substantial
knowledge/
intellectual
property/
important
innovations.

Several Systems with very high
protection need affected, more
than 10.000 accounts or
individual systems affected,
Unstopped self-spreading
malware (Ransomware) active.

External reporting to
governmental entities required.

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
9/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

Severity 2
(High)

very high level of
effort.

Operational
processes are
disrupted.
Transition to
replacement
processes
requires a great
deal of effort.

Severity 1
(Medium)

Severity 0
(Low)

Operational
processes are
disrupted.
Disruptions are
not tolerable.
Transition to
replacement
processes
requires
considerable
effort.

Operational
processes are
minimally
disrupted.
Disruptions are
tolerable.

Significantly
perceptible
negative effects
over a longer
term.

Loss of
significant
knowledge/
intellectual
property/
innovations.

At least one System with very
high protection needs partially
affected (e.g. due to impact from
other, affected systems), less
than 10.000 accounts or
individual systems affected, self
spreading malware detected but
spreading stopped.

External reporting to
governmental entities required.

Perceptible
negative effects
in the short-term.
.

Loss of
important
knowledge/
intellectual
property.

Less than 1000 accounts or
individual systems affected, blast
radius of incident known and
effects are under control.

No external reporting to
governmental entities required.

Little or no
perceptible
negative effects.

No loss of
knowledge.

Less than 100 accounts or
individual systems affected.

No external reporting to
governmental entities required.

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
10/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

3.6

Information Sensitivity

According to C/ISP CD 02900 Information security incidents and the knowledge obtained from
them must be classified as security class C-SC3 or I-SC3. The release of this information outside
the Bosch Group must be agreed with C/ISP and C/CYG in advance. The need-to-know principle
must be followed.

Bosch Cybersecurity Incident Management utilizes an adapted version of the Traffic Light
Protocol (TLP1) which can be found here

Colour

Description

Example

Red

Recipients must not share TLP:RED
information with any parties outside of
the specific exchange, meeting, or
conversation in which it was originally
disclosed. Only the author of the
information is able to grant exceptions.

Amber

Ambe+Strict

TLP:AMBER is the standard
classification of information. Information
without a specific classification have to
be handled according this category.

Recipients can only share parts of
TLP:AMBER information with members
of their own organization or 3rd Party
(e.g. external operators with NDA) ac-
cording the „Need-to-Know“-Principle .
So that the recipients, as long as they
are working to protect the IT Security in
the own company, are able to
implement appropriate measures for
Detection or Protection.

TLP:RED information about a
current ongoing cyber-attack in a
face-to-face meeting MUST NOT
be shared. A participant of this
meeting is not allowed to share
this information with anybody. Also
not with colleagues in the team or
with any superior also not
correspondingly.

Example: Bosch CERT receives
an information category
TLP:AMBER, e.g. an IP-address,
and instructs thereupon the IT-
Service Provider (BD), to block
this IP. According the „Need-to-
Know“-Principle BD only needs to
know the IP-address, but not the
context or reason why this IP has
to be blocked. It is strictly
forbidden to hand over
TLP:AMBER-Information to further
employees or customer (e.g., of
ETAS, BSH) of the Bosch-Group,
also not correspondingly.

1 https://www.us-cert.gov/tlp

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
11/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

TLP:AMBER+STRICT limits the
sharing to only the own organisation,
so in this case, no 3rd Party may
receive the information

Recipients may share TLP:GREEN
information with peers and partner
organizations within their sector or
community, but not via publicly
accessible channels.

TLP:AMBER+STRICT information
needs to stay inside the informed
organisation, so in the example
above, the IP address is not
allowed to be passed to a 3rd Party
outside BD.

Bosch CERT sends a specific
security notification to a sector (1-
to-many, limited)

Subject to standard copyright rules,
TLP:WHITE information may be
distributed without restriction.

Public security advisory or
notification published on the
Internet (1-to-any, unlimited)

Green

Clear

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
12/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

4  Cybersecurity Incident Response Lifecycle

Bosch has a distinct incident response lifecycle2 that is used to guide IT Owners and IT Users in
effectively detecting, responding, and containing cybersecurity incidents. The lifecycle is agile to
be able to adapt to the various types of cybersecurity threats targeting the Bosch Group.

2 https://nvlpubs.nist.gov/nistpubs/specialpublications/nist.sp.800-61r2.pdf

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
13/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

4.1  Prepare

Incident response needs to establish incident response capability so that the organization is ready
to respond to incidents.

Description

Develop and Update the Cybersecurity Incident Response Plan:

Make any necessary updates of this document.

Responsible,
Support

C/CYG, C/ISP,
Bosch CERT

Develop and Update Cybersecurity Incident Reporting Form:

Bosch CERT, C/CYG

Based on the results from previous incidents and their corresponding Lessons
Learnt workshops, update the supporting plans to improve the efficiency and
effectiveness of the incident response process. Provide a reporting form for
incident reporting entities to contact Bosch CERT effectively.

Develop and Update Standard Operation Procedures (SOPs) & Playbooks:

Make any necessary updates, to response tasks, containment actions, and
workflow. If the Incident does not fit to the SOPs already described and it seems
to be an incident that could occur repeatedly, create a new SOP and specify how
to respond to that incident type.

Bosch CERT,
C/CYG, C/ISP

Identify New Detection Logic:

If the incident was not detected through an alert and/or it was not detected initially
by an alert, consider developing new detection logic to improve the time‐to‐detect
in the future.

Bosch CERT,
Support by IT Owner

Identify, Update and Implement New Security Regulations:

If the incident was caused because of missing regulation or vulnerable measures,
consider developing new regulations.

C/CYG, C/ISP,
Bosch CERT, IT
Owner

4.2  Detection, & Analysis

This phase of the incident response lifecycle covers the detection and first analysis (Triage) to
determine a cybersecurity incident.

It covers the process from cybersecurity event to an incident by

•  analyzing the impacted environment

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
14/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

•  determine the Attack Vector
•  determine the Security Incident Classification
•  determine the Severity rating to prioritize the incident.
•  determine an Incident Handling Officer
•  determine Information Sensitivity - Traffic Light Protocol (TLP) rating.
•  determine the business impact.
•  gather incident details.

4.3  Containment, Eradication, and Recovery

Containment is important before an incident overwhelms resources or increases damage.

This phase covers:

•  Develop Containment, Eradication, and Recovery strategy.
•

Identify Response Tasks

After an incident has been contained, eradication may be necessary to eliminate components of
the incident, such as deleting malware and disabling breached user accounts, as well as
identifying and mitigating all vulnerabilities that were exploited. During eradication, it is important
that all identified and affected hosts within the organization are remediated.

In recovery, administrators restore systems to normal operation, confirm that the systems are
functioning normally, and (if applicable) remediate vulnerabilities to prevent similar incidents.
Recovery may involve such actions as restoring systems from clean backups, rebuilding systems
from scratch, replacing compromised files with clean versions, installing patches, changing
passwords, and tightening network perimeter security (e.g., firewall rulesets, boundary router
access control lists). Higher levels of system logging or network monitoring are often part of the
recovery process. Once a resource is successfully attacked, it is often attacked again, or other
resources within the organization are attacked in a similar manner.

4.4  Post-Incident Activity

In order to assess the quality of incident response and ensure all corrective actions are identified
the Incident Handling Officer has to perform a post-incident Lessons Learnt workshop and
compile a postmortem report at least for incidents with critical and high severity before closing the
incident. The prepared postmortem reports are available C/CY-internally on request to increase
the cybersecurity resilience and enable the improvement of the PDCA-Cycle. Entities outside of
C/CY may get access to individual postmortem reports on a need-to-know basis. The
requirements from Chapter 2.7.1 Lessons Learned from Cybersecurity Incidents of CD 09000
apply.

The Post‐Incident review should answer the following questions:

•  Have all corrective actions been identified?
•  Do they provide sufficient detail, and are they assigned correctly?

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
15/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

•  Was a postmortem report created?
•  How well was the incident response process followed?

5  Cybersecurity Response Activities

The Incident Handling Officer is responsible for coordinating all cybersecurity incident response
activities. In order to do so, the Incident Handling Officer gets in contact with different
stakeholders. The following section depicts the interfaces towards stakeholders outside of C/CY
and defines the required cooperation as well as acceptable response times.

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
16/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
17/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

Phase

Interface  Responsible  Communication

Description

Timing

Form

Analysis

I0

Incident
Reporting
Entity

partner

Bosch CERT

Incident Identification/Detection:

immediately

Entity reports a potential
cybersecurity incident to Bosch
CERT.

OR

Bosch IT security monitoring
mechanisms detect potential
security incident and report to
Bosch CERT

Email, Security
Monitoring Console,
Reporting Form

Bosch CERT

Incident
Reporting Entity

Bosch CERT confirms and
registers security event.

Within one working
day

Email

Triage

I0

Bosch CERT

Incident
Reporting Entity

In case the Triage can rule out an
cybersecurity incident, the Incident
Reporting Entity is notified. In any
other case, an Incident Handling

Within one working
day

Email, phone call

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
18/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

I1

Incident
Handling
Officer

Relevant
Stakeholders

Analysis

I2

C/CYT

Incident
Handling
Officer

Officer is defined, who is leading
the handling of the cybersecurity
incident.

After an initial assessment of the
cybersecurity incident, all relevant
stakeholders are informed. The
relevance is derived from the
severity of the incident, the need
for collaboration and details of the
case, e.g. the responsible OE ISP
Office is informed in case of an
information protection incident.  If
required by law governance
agencies are informed within the
required time span.

If the further analysis reveals a
critical cybersecurity incident.
C/CYT is directly notified about the
situation for further decision on
escalation and execution of the
crisis management process.

Email, Telco

Depending on
Severity.
3- one day
2- one workday
1- regular reporting
0 - if necessary

immediately

Email & Telco

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
19/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

Containm

I3

ent,

Erradicati

on &

Recovery

I4

Closing

I5

Incident
Handling
Officer

Incident
Handling
Officer

Incident
Handling
Officer

Incident
Handling
Officer

C/CYT, C/CY

Asset Owner,
Asset Operator,
OE CSO &
Cybersecurity
Organization

All relevant
stakeholders

All relevant
stakeholders

Depending on the severity of a
case, the Incident Handling Officer
reports to C/CYT and C/CY on a
regular basis.

The Incident Handling Officer
involves the IT Owner, the OE
CSO and his cybersecurity
organization to contain, eradicate,
and recover the cybersecurity
incident.

The Incident Handling Officer
involves all relevant parties to
contribute to lessons learned.

The Incident Handling Officer
informs all entities, which are
directly or indirectly affected by the
measures defined in the lessons
learned, about the content of the
lessons learned as well as all
measure to be implemented.

Depending on
Severity.

Email & Telco

Within one working
day

Email & Telco

Within 10 working
days

Email & Telco

Within 10 working
days

Email

Valid issue only in Intranet. No change service for print-outs.

Corporate Good Practice

Issue/Amendment
1.7

Page
20/20

From
C/CYG

Cybersecurity Incident Response Plan

Our Reference
C/CYG

Date
Jan. 1, 2025

6  Appendix

6.1

Incident Reporting Form

Cybersecurity Incidents must be reported to “Bosch CERT” by email to CERT@bosch.com”.

Also is a reporting form in the intranet available, Link: https://bgn.bosch.com/alias/security-
incident

6.2  Obligations

Associates and involved third parties have to sign the Obligation of declaration (aka “C/CY Code
of Conduct”), which can be provided by C/CYG to commit to handle any information they get
access to during their work (verbal, written or any other way) according to its classification. Other
Bosch guidelines and Bosch declaration of obligation remain unaffected.

7  Revision History

Issue  Date

Editor

Description of amendment

1.0

01/14/2019  C/IDS

Initial publication

Rüping

1.1

11/29/2019  C/IDS

Rüping

Change Bosch crisis Team into C Crisis Management
Team. Update Links

1.2

07/15/2020  C/IDS
Grau,
-
Rüping,
C/IDS-
GE

Change wording from Information Security Incident to
Cybersecurity Incident, Add Security Incident
Classification, Review and minor changes

1.3

01/02/2021  C/IDS-

Rename of CGP

CD

1.4

1.5

1.7

10/02/2023  C/CYG  Organizational Changes regarding BD and C/CY;
Changes of TLP 2.0 integrated (AMBER+STRICT,
CLEAR); Alignment of definitions to updated
regulations

04/26/2023  C/CYG  Correction in Process Diagram 5.1

01/01/2025  C/CYG  Rework to fit to Re-Org & CD 09000

Valid issue only in Intranet. No change service for print-outs.



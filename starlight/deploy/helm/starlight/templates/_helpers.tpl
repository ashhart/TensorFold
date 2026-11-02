{{- define "starlight.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "starlight.fullname" -}}
{{- if contains .Chart.Name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "starlight.labels" -}}
app.kubernetes.io/name: {{ include "starlight.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "starlight.selector" -}}
app.kubernetes.io/name: {{ include "starlight.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "starlight.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}
{{- end -}}

{{- define "starlight.serviceAccount" -}}
{{- if .Values.serviceAccount.create -}}
{{ .Values.serviceAccount.name | default (include "starlight.fullname" .) }}
{{- else -}}
{{ .Values.serviceAccount.name | default "default" }}
{{- end -}}
{{- end -}}

{{- define "starlight.secretEnv" -}}
- name: STARLIGHT_DATABASE_URL_FILE
  value: /run/secrets/starlight/database-url
- name: STARLIGHT_MASTER_KEY_FILE
  value: /run/secrets/starlight/master-key
{{- end -}}

{{- define "starlight.secretVolume" -}}
- name: secrets
  secret:
    secretName: {{ .Values.secrets.name }}
    defaultMode: 0400
    items:
      - key: {{ .Values.secrets.databaseURLKey }}
        path: database-url
      - key: {{ .Values.secrets.masterKeyKey }}
        path: master-key
      {{- if .Values.secrets.metricsTokenKey }}
      - key: {{ .Values.secrets.metricsTokenKey }}
        path: metrics-token
      {{- end }}
{{- end -}}

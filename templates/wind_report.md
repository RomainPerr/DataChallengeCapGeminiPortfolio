# {{ report_title }}

**Type d’actif :** {{ asset_label }} · **Généré le :** {{ generation_date }}

## Synopsis

> {{ synopsis }}

---

## 1. Chiffres clés

| Indicateur | Valeur |
| --- | ---: |
| Nombre d’incidents analysés | {{ n_incidents }} |
| Nombre de sites couverts | {{ n_sites }} |
| Nombre d’éoliennes surveillées | {{ n_equipment }} |
| Période couverte | {{ period_start }} → {{ period_end }} |
| Coût total estimé | {{ total_cost_eur }} EUR |
| Durée totale d’indisponibilité | {{ total_downtime_hours }} h |
| Coût moyen par incident | {{ avg_cost_per_incident }} EUR |
| Coût médian par incident | {{ median_cost_per_incident }} EUR |
| Durée moyenne d’indisponibilité par incident | {{ avg_downtime_per_incident }} h |
| Durée médiane d’indisponibilité par incident | {{ median_downtime_per_incident }} h |
| Coût moyen par heure d’indisponibilité | {{ avg_cost_per_downtime_hour }} EUR/h |

**Répartition par niveau de sévérité :** {{ n_minor }} mineurs · {{ n_moderate }} modérés · {{ n_severe }} sévères

**Site le plus coûteux :** {{ top_site_by_cost }} — **Éolienne la plus coûteuse :** {{ top_equipment_by_cost }}

{{ figure_plot_yearly_incidents }}

*Évolution annuelle du nombre d’incidents et des coûts totaux estimés.*

{{ figure_plot_yearly_variation }}

*Variation annuelle des coûts des incidents et de la durée d’indisponibilité.*

## 2. Analyse des coûts

{{ methodology }}

### 2.1 Coût par site

{{ figure_plot_total_cost_by_site }}

*Coût total estimé par parc éolien, ventilé par type d’actif.*

### 2.2 Distribution des coûts par éolienne

{{ figure_plot_cost_distribution_per_equipment_by_site }}

*Distribution du coût total estimé par éolienne, regroupée par site.*

### 2.3 Coût quotidien de l’éolienne la plus impactée

{{ figure_plot_daily_cost_for_equipment }}

*Évolution quotidienne du coût estimé pour l’éolienne présentant le coût total le plus élevé.*

### 2.4 Principales éoliennes du site le plus coûteux

{{ figure_plot_daily_cost_top_equipment_at_top_site }}

*Coût quotidien estimé des trois éoliennes les plus coûteuses du site présentant le coût total le plus élevé.*

### 2.5 Indisponibilité et coût

{{ figure_plot_downtime_vs_cost }}

*Coût estimé en fonction de la durée d’indisponibilité, avec une distinction par niveau de sévérité.*

## 3. Exposition aux conditions météorologiques

Les éoliennes sont directement exposées aux conditions météorologiques. Les graphiques ci-dessous mettent en relation la dynamique des incidents et leurs coûts avec les conditions météorologiques enregistrées au moment de chaque événement.

### 3.1 Coût par condition météorologique

{{ figure_plot_total_cost_by_weather_condition }}

*Coût total estimé par condition météorologique, ventilé par type d’actif — les conditions liées aux tempêtes se distinguent notamment.*

### 3.2 Nombre mensuel d’incidents

{{ figure_plot_monthly_incident_count }}

*Nombre d’incidents par mois sur la période étudiée, ventilé par type d’actif.*

### 3.3 Incidents par niveau de sévérité

{{ figure_plot_incident_count_by_severity }}

*Nombre d’incidents par niveau de sévérité, ventilé par type d’actif.*

## 4. Clusters thématiques

{{ n_clusters }} groupes thématiques d’incidents (*clusters*) ont été identifiés à partir des descriptions des rapports.

{{ top_clusters_table }}

{{ cluster_sections }}

## 5. Recommandations

{{ recommendations }}

## 6. Synthèse

{{ synthesis }}
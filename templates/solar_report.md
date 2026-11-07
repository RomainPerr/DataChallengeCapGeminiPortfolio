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
| Nombre d’équipements surveillés | {{ n_equipment }} |
| Période couverte | {{ period_start }} → {{ period_end }} |
| Coût total estimé | {{ total_cost_eur }} EUR |
| Durée totale d’indisponibilité | {{ total_downtime_hours }} h |
| Coût moyen par incident | {{ avg_cost_per_incident }} EUR |
| Coût médian par incident | {{ median_cost_per_incident }} EUR |
| Durée moyenne d’indisponibilité par incident | {{ avg_downtime_per_incident }} h |
| Durée médiane d’indisponibilité par incident | {{ median_downtime_per_incident }} h |
| Coût moyen par heure d’indisponibilité | {{ avg_cost_per_downtime_hour }} EUR/h |

**Répartition par niveau de sévérité :** {{ n_minor }} mineurs · {{ n_moderate }} modérés · {{ n_severe }} sévères

**Site le plus coûteux :** {{ top_site_by_cost }} — **Équipement le plus coûteux :** {{ top_equipment_by_cost }}

{{ figure_plot_yearly_incidents }}

*Évolution annuelle du nombre d’incidents et des coûts totaux estimés.*

{{ figure_plot_yearly_variation }}

*Variation annuelle des coûts des incidents et de la durée d’indisponibilité.*

## 2. Analyse des coûts

{{ methodology }}

### 2.1 Coût par site

{{ figure_plot_total_cost_by_site }}

*Coût total estimé (barres rouges) et nombre d’incidents (courbe bleue) par site.*

{{ interpretation_plot_total_cost_by_site }}

### 2.2 Distribution des coûts par équipement

{{ figure_plot_cost_distribution_per_equipment_by_site }}

*Distribution du coût total estimé par équipement, regroupée par site.*

### 2.3 Top 3 des équipements les plus coûteux du site le plus impacté

{{ figure_plot_daily_cost_top_equipment_at_top_site }}

*Évolution quotidienne du coût estimé des trois équipements les plus coûteux du site présentant le coût total estimé le plus élevé.*

## 3. Analyse temporelle et par niveau de sévérité

### 3.1 Nombre d’incidents et coût par mois

{{ figure_plot_monthly_incident_count }}

*Nombre d’incidents (barres bleues) et coût total estimé (courbe rouge) par mois sur la période étudiée.*

{{ interpretation_plot_monthly_incident_count }}

### 3.2 Nombre d’incidents et coût par niveau de sévérité

{{ figure_plot_incident_count_by_severity }}

*Nombre d’incidents (barres bleues) et coût total estimé (courbe rouge) par niveau de sévérité.*

{{ interpretation_plot_incident_count_by_severity }}

### 3.3 Conditions météorologiques

{{ figure_plot_total_cost_by_weather_condition }}

*Coût total estimé (barres rouges) et nombre d’incidents (courbe bleue) selon les conditions météorologiques.*

{{ interpretation_plot_total_cost_by_weather_condition }}

## 4. Clusters thématiques

{{ n_clusters }} groupes thématiques d’incidents (*clusters*) ont été identifiés à partir des descriptions des rapports.

{{ top_clusters_table }}

### 4.1 Fréquence et coût par cluster

{{ figure_plot_cluster_frequency_cost }}

*Nombre d’incidents (barres bleues) et coût total estimé (courbe rouge) par cluster thématique.*

{{ interpretation_plot_cluster_frequency_cost }}

{{ cluster_sections }}

## 5. Recommandations

{{ recommendations }}

## 6. Synthèse

{{ synthesis }}
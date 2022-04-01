import osm2gmns as og

net = og.getNetFromFile('data/osm_data/shenzhen.osm', link_types=('motorway', 'trunk', 'primary', 'secondary', 'tertiary'))
og.outputNetToCSV(net, output_folder="data/osm_data/export_data/new1")

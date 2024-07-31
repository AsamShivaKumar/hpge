import re

def writeYearWise(file_path):
  file = open(file_path, encoding="utf8")
  year_file_map = {}
  nodes = {}
  nodes['dummy_node'] = 0
  line = file.readline()
  cnt = 0

  while True:
      if not line: break
      if cnt > 0: line = file.readline()
      cnt += 1

      authors = []
      year = ''
      conf = ''
      edge_list = []

      for i in range(7):
        if i == 1: authors = line[2:-1].split(',')
        if i == 2: year = line[5:-1]
        if i == 3: conf = line[5:-1]
        line = file.readline()

      while line and not(line == '\n'): line = file.readline()

      if nodes.get(conf) == None: nodes[conf] = len(nodes.keys())
      conf_id = nodes.get(conf)

      author_ids = []
      for author in authors:
          if nodes.get(author) == None: nodes[author] = len(nodes.keys())
          author_id = nodes[author]
          author_ids.append(author_id)
          
          edge_list.append([str(author_id),'0',str(conf_id),'1','0',year])
          edge_list.append([str(conf_id),'1',str(author_id),'0','2',year])  #******************

      for i in author_ids:
          for j in author_ids:
              if i >= j: continue
              edge_list.append([str(i),'0',str(j),'0','1',year])

      # write edges
      if len(year) < 4: continue
      if year_file_map.get(year) == None: year_file_map[year] = open('data/year_wise/' + year, 'w+')
      writer = year_file_map[year]
      for edge in edge_list:
        writer.write(','.join(edge))
        writer.write('\n')

      
import numpy
import pandas

"""
Parse a GENCODE GTF file and return a dictionary of gene data.
"""


def parse_gtf_file(file_path):
    gene_data = {}  # Dictionary to store parsed data

    with open(file_path, 'r') as gtf_file:
        for line in gtf_file:
            if line.startswith('#'):
                continue  # Skip comments

            parts = line.strip().split('\t')
            if len(parts) != 9:
                continue  # Skip invalid lines

            feature_type = parts[2]
            strand = parts[6]

            if feature_type == 'gene':
                gene_id = None
                gene_name = None

                attributes = parts[8].split('; ')
                for attribute in attributes:
                    key, value = attribute.split(' ')
                    if key == 'gene_id':
                        gene_id = value.strip('";').split(".")[0]  # remove the '.XX' versioning, keep only the prefix
                    elif key == 'gene_name':
                        gene_name = value.strip('";')
                    elif key == 'gene_type':
                        gene_type = value.strip('";')

                if gene_id:
                    gene_data[gene_id] = {
                        'gene_name': gene_name,
                        'gene_type': gene_type,
                        'transcripts': []
                    }

            elif feature_type == 'transcript':
                gene_id = None
                transcript_id = None

                if strand == '+':
                    transcript_start = int(parts[3])
                    transcript_end = int(parts[4])

                elif strand == '-':
                    transcript_start = int(parts[4])
                    transcript_end = int(parts[3])

                attributes = parts[8].split('; ')
                for attribute in attributes:
                    key, value = attribute.split(' ')
                    if key == 'gene_id':
                        gene_id = value.strip('";').split(".")[0]  # remove the '.XX' versioning
                    elif key == 'transcript_id':
                        transcript_id = value.strip('";')
                    elif key == 'transcript_type':
                        transcript_type = value.strip('";')

                if gene_id and transcript_id:
                    gene_data[gene_id]['transcripts'].append({
                        'transcript_id': transcript_id,
                        'transcript_type': transcript_type,
                        'start': transcript_start,
                        'end': transcript_end,
                        'strand': strand
                    })
    return gene_data



/* Adapted from 
 * http://rosettacode.org/wiki/Bitmap/Read_a_PPM_file#C
 */

#ifndef HEADER_SEEN_PPM_H
#define HEADER_SEEN_PPM_H

#define PPMREADBUFLEN 256

#include <stddef.h>

typedef unsigned char channel_t;
typedef struct
{
  size_t width, height;
  channel_t *red_buffer, *green_buffer, *blue_buffer;
} image_t;

image_t *alloc_image(size_t w, size_t h);
void free_image(image_t *img);
image_t *read_ppm(char const *file_name);
int write_ppm(char const *file_name, const image_t *img);

#endif

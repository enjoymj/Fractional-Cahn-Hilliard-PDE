#include "ppm.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

image_t *alloc_image(size_t w, size_t h)
{
  image_t *result = (image_t *) malloc(sizeof(image_t));

  if (!result)
  {
    fprintf(stderr, "failed to allocate image\n");
    return NULL;
  }

  result->width = w;
  result->height = h;

  result->red_buffer = malloc(sizeof(channel_t)*w*h);
  result->green_buffer = malloc(sizeof(channel_t)*w*h);
  result->blue_buffer = malloc(sizeof(channel_t)*w*h);
  if (!(result->red_buffer && result->green_buffer && result->blue_buffer))
  {
    fprintf(stderr, "failed to allocate image buffer\n");
    free(result->red_buffer);
    free(result->green_buffer);
    free(result->blue_buffer);
    free(result);
    return NULL;
  }
  return result;
}

void free_image(image_t *img)
{
  free(img->red_buffer);
  free(img->green_buffer);
  free(img->blue_buffer);
  free(img);
}

image_t *read_ppm(char const *file_name)
{
  char buf[PPMREADBUFLEN], *t;
  int r;

  FILE *file = fopen(file_name, "r");
  if (file == NULL)
  {
    perror("opening file for reading");
    return NULL;
  }

  t = fgets(buf, PPMREADBUFLEN, file);
  /* the code fails if the white space following "P6" is not '\n' */
  if (t == NULL)
  {
    fprintf(stderr, "unexpected end of file\n");
    fclose(file);
    return NULL;
  }
  if (strncmp(buf, "P6\n", 3) != 0)
  {
    fprintf(stderr, "unexpected magic number\n");
    fclose(file);
    return NULL;
  }

  do
  { /* Px formats can have # comments after first line */
    t = fgets(buf, PPMREADBUFLEN, file);
    if ( t == NULL )
    {
      fprintf(stderr, "unexpected end of file\n");
      fclose(file);
      return NULL;
    }
  }
  while ( strncmp(buf, "#", 1) == 0 );

  unsigned int w, h;
  r = sscanf(buf, "%u %u", &w, &h);
  if (r < 2)
  {
    fprintf(stderr, "unexpected end of file\n");
    fclose(file);
    return NULL;
  }

  unsigned int d;
  r = fscanf(file, "%u", &d);
  if (r < 1)
  {
    fprintf(stderr, "unexpected end of file\n");
    fclose(file);
    return NULL;
  }

  if (d != 255)
  {
    fprintf(stderr, "only 8-bit images supported\n");
    fclose(file);
    return NULL;
  }
  fseek(file, 1, SEEK_CUR); /* skip one byte, should be whitespace */

  size_t bufsize = 3*w*h;
  channel_t *temp_buffer = malloc(sizeof(channel_t)*bufsize);
  if (!temp_buffer)
  {
    fprintf(stderr, "failed to allocate temp buffer\n");
    fclose(file);
    return NULL;
  }

  size_t rd = fread(temp_buffer, sizeof(channel_t), bufsize, file);
  if (rd < w*h)
  {
    fprintf(stderr, "unexpected end of file\n");
    free(temp_buffer);
    fclose(file);
    return NULL;
  }

  image_t *img = alloc_image(w, h);
  if (img == NULL)
  {
    free(temp_buffer);
    free(temp_buffer);
    fclose(file);
    return NULL;
  }

  for (size_t i = 0; i<w*h; ++i)
  {
    img->red_buffer[i] = temp_buffer[3*i];
    img->green_buffer[i] = temp_buffer[3*i+1];
    img->blue_buffer[i] = temp_buffer[3*i+2];
  }

  free(temp_buffer);
  fclose(file);
  return img;
}

int write_ppm(char const *file_name, const image_t *img)
{
  size_t bufsize = 3*img->width*img->height;
  channel_t *temp_buffer = malloc(sizeof(channel_t)*bufsize);
  if (!temp_buffer)
  {
    fprintf(stderr, "failed to allocate temp buffer\n");
    return -1;
  }

  for (size_t i = 0; i<img->width*img->height; ++i)
  {
    temp_buffer[3*i] = img->red_buffer[i];
    temp_buffer[3*i+1] = img->green_buffer[i];
    temp_buffer[3*i+2] = img->blue_buffer[i];
  }

  FILE *file = fopen(file_name, "w");
  if (file == NULL)
  {
    perror("opening file for writing");
    return -1;
  }
  fprintf(file, "P6\n%d %d\n255\n", img->width, img->height);

  size_t write_size = 3*img->width*img->height;
  size_t written = fwrite(temp_buffer,
      sizeof(channel_t), write_size, file);
  if (written < write_size)
  {
    perror("writing image buffer");
    fclose(file);
    free(temp_buffer);
    return -1;
  }

  fclose(file);
  free(temp_buffer);
}

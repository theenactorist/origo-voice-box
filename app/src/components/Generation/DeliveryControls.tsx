import type { UseFormReturn } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import {
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { Textarea } from '@/components/ui/textarea';
import { DELIVERY_PRESETS } from '@/lib/constants/delivery';
import type { GenerationFormValues } from '@/lib/hooks/useGenerationForm';

interface DeliveryControlsProps {
  form: UseFormReturn<GenerationFormValues>;
}

export function DeliveryControls({ form }: DeliveryControlsProps) {
  const { t } = useTranslation();

  return (
    <FormField
      control={form.control}
      name="instruct"
      render={({ field }) => (
        <FormItem className="mt-2 rounded-2xl border border-accent/20 p-3">
          <FormLabel className="text-xs font-medium">{t('generation.instruct.label')}</FormLabel>
          <fieldset
            aria-label={t('generation.instruct.presetsLabel')}
            className="flex flex-wrap gap-1.5"
          >
            {DELIVERY_PRESETS.map((preset) => (
              <Button
                key={preset.id}
                type="button"
                size="sm"
                variant={field.value === preset.instruct ? 'secondary' : 'outline'}
                className="h-7 rounded-full px-3 text-xs"
                aria-pressed={field.value === preset.instruct}
                onClick={() => field.onChange(preset.instruct)}
              >
                {t(`generation.instruct.presets.${preset.id}`)}
              </Button>
            ))}
          </fieldset>
          <FormControl>
            <Textarea
              {...field}
              value={field.value ?? ''}
              placeholder={t('generation.instruct.placeholder')}
              className="min-h-[60px] max-h-40 resize-y rounded-xl bg-transparent px-3 py-2 text-sm placeholder:text-muted-foreground/60"
              maxLength={500}
            />
          </FormControl>
          <FormDescription className="text-xs">
            {t('generation.instruct.description')}
          </FormDescription>
          <FormMessage className="text-xs" />
        </FormItem>
      )}
    />
  );
}

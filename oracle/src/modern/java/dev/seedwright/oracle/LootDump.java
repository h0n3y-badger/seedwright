package dev.seedwright.oracle;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Function;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Holder;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.ResourceKey;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.RandomizableContainer;
import net.minecraft.world.SimpleContainer;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.ChunkGeneratorStructureState;
import net.minecraft.world.level.chunk.status.ChunkStatus;
import net.minecraft.world.level.levelgen.structure.BoundingBox;
import net.minecraft.world.level.levelgen.structure.Structure;
import net.minecraft.world.level.levelgen.structure.StructureSet;
import net.minecraft.world.level.levelgen.structure.StructureStart;
import net.minecraft.world.level.storage.loot.LootParams;
import net.minecraft.world.level.storage.loot.LootTable;
import net.minecraft.world.level.storage.loot.parameters.LootContextParamSets;
import net.minecraft.world.level.storage.loot.parameters.LootContextParams;
import net.minecraft.world.phys.Vec3;

/**
 * Chest loot ground truth (26.3+): -Dseedwright.oracle.loot=R. For every start of the village and buried treasure
 * sets within R chunks, generates the chunks it covers to FULL and lists each loot container inside its bounding
 * box: position, loot table, LootTableSeed and the item totals the table rolls with that seed (no luck), as the
 * container would get them when first opened. Also the structure registry order per generation step (the index
 * that goes into WorldgenRandom.setFeatureSeed). Written to loot-&lt;seed&gt;.txt.
 */
public final class LootDump
{
	private LootDump() {}

	public static void dump(ServerLevel level, ChunkGeneratorStructureState state, long seed, int radius, Function<Holder<StructureSet>, String> setName,
		Map<Structure, String> structureNames) throws Exception
	{
		StringBuilder out = new StringBuilder("seed ").append(seed).append('\n');
		var reg = level.registryAccess().lookupOrThrow(Registries.STRUCTURE);
		Map<Integer, Integer> perStep = new TreeMap<>();
		for (Structure s : (Iterable<Structure>) reg.stream()::iterator)
		{
			int step = s.step().ordinal();
			int idx = perStep.merge(step, 1, Integer::sum) - 1;
			out.append("order ").append(reg.getKey(s)).append(' ').append(step).append(' ').append(idx).append('\n');
		}
		List<String[]> todo = new ArrayList<>();
		int containers = 0;
		for (Holder<StructureSet> set : state.possibleStructureSets())
		{
			String sn = setName.apply(set);
			if (!sn.endsWith(":villages") && !sn.endsWith(":buried_treasures"))
			{
				continue;
			}
			for (int x = -radius; x <= radius; x++)
			{
				for (int z = -radius; z <= radius; z++)
				{
					if (!set.value().placement().isStructureChunk(state, x, z))
					{
						continue;
					}
					ChunkAccess startChunk = level.getChunk(x, z, ChunkStatus.STRUCTURE_STARTS, true);
					for (Map.Entry<Structure, StructureStart> e : startChunk.getAllStarts().entrySet())
					{
						StructureStart start = e.getValue();
						boolean inSet = false;
						for (StructureSet.StructureSelectionEntry se : set.value().structures())
						{
							inSet |= se.structure().value() == e.getKey();
						}
						if (!inSet || !start.isValid())
						{
							continue;
						}
						String name = structureNames.getOrDefault(e.getKey(), "?");
						BoundingBox b = start.getBoundingBox();
						out.append("structure ").append(name).append(' ').append(x).append(' ').append(z).append('\n');
						for (int cx = b.minX() >> 4; cx <= b.maxX() >> 4; cx++)
						{
							for (int cz = b.minZ() >> 4; cz <= b.maxZ() >> 4; cz++)
							{
								ChunkAccess c = level.getChunk(cx, cz, ChunkStatus.FULL, true);
								List<BlockPos> sorted = new ArrayList<>(c.getBlockEntitiesPos());
								sorted.sort((p, q) -> p.getY() != q.getY() ? p.getY() - q.getY() : p.getX() != q.getX() ? p.getX() - q.getX() : p.getZ() - q.getZ());
								for (BlockPos pos : sorted)
								{
									BlockEntity be = c.getBlockEntity(pos);
									if (!(be instanceof RandomizableContainer rc) || pos.getX() < b.minX() || pos.getX() > b.maxX() || pos.getZ() < b.minZ() || pos.getZ() > b.maxZ())  // x/z only: a buried treasure start box is one block at y=90
									{
										continue;
									}
									ResourceKey<LootTable> key = rc.getLootTable();
									out.append("container ").append(pos.getX()).append(' ').append(pos.getY()).append(' ').append(pos.getZ()).append(' ')
										.append(key == null ? "none" : key.identifier().toString()).append(' ').append(rc.getLootTableSeed());
									if (key != null)
									{
										LootTable table = level.getServer().reloadableRegistries().getLootTable(key);
										SimpleContainer inv = new SimpleContainer(27);
										LootParams params = new LootParams.Builder(level).withParameter(LootContextParams.ORIGIN, Vec3.atCenterOf(pos))
											.create(LootContextParamSets.CHEST);
										table.fill(inv, params, rc.getLootTableSeed());
										Map<String, Integer> tot = new TreeMap<>();
										for (int i = 0; i < inv.getContainerSize(); i++)
										{
											ItemStack st = inv.getItem(i);
											if (!st.isEmpty())
											{
												tot.merge(BuiltInRegistries.ITEM.getKey(st.getItem()).toString(), st.getCount(), Integer::sum);
											}
										}
										for (Map.Entry<String, Integer> t : tot.entrySet())
										{
											out.append(' ').append(t.getKey()).append('=').append(t.getValue());
										}
									}
									out.append('\n');
									containers++;
								}
							}
						}
					}
				}
			}
		}
		out.append("end ").append(containers).append('\n');
		Files.writeString(Path.of("loot-" + seed + ".txt"), out);
	}
}

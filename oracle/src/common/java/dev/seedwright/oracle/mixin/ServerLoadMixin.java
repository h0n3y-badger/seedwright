package dev.seedwright.oracle.mixin;

import dev.seedwright.oracle.Oracle;
import net.minecraft.server.MinecraftServer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(MinecraftServer.class)
public class ServerLoadMixin
{
	@Inject(method = "loadLevel", at = @At("TAIL"))
	private void seedwright$afterLoad(CallbackInfo ci)
	{
		Oracle.run((MinecraftServer) (Object) this);
	}
}
